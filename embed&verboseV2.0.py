import os
import json
import time
import re
import numpy as np
import pymupdf  # fitz
from typing import TypedDict, List, Dict, Annotated
import operator
from scipy.optimize import linear_sum_assignment

# Библиотеки AI
try:
    from sentence_transformers import SentenceTransformer, util
    from langgraph.graph import StateGraph, END
    import torch
except ImportError:
    print(
        "❌ ОШИБКА: Установите: pip install sentence-transformers langgraph scipy torch"
    )
    exit()

from llama_cpp import Llama

# ========== ⚙️ КОНФИГУРАЦИЯ ==========

#PATH_TO_LLM = r"/home/seral/diploma/dev_1_conda/models/qwen-14b/Qwen2.5-14B-Instruct-Q4_K_M.gguf"
PATH_TO_LLM = r"/home/seral/diploma/dev_1_conda/models/llama-abliterated/Meta-Llama-3.1-8B-Instruct-abliterated-Q4_K_M.gguf"
PATH_TO_LABSE = "./models_cache/LaBSE"

# Пороги классификации
THRESHOLD_IDENTICAL = 0.98
THRESHOLD_MODIFIED = 0.72

# Настройки железа
N_GPU_LAYERS = 50
N_CTX = 8192

# Батчинг для больших документов (критично!)
BATCH_SIZE_SIMILARITY = 100

# Ключевые слова для приоритизации
CRITICAL_KEYWORDS = [
    "обязан",
    "вправе",
    "запрещено",
    "штраф",
    "санкция",
    "ответственность",
    "срок",
    "дней",
    "месяцев",
    "рублей",
    "процент",
    "должен",
    "может",
    "не может",
    "не вправе",
]

MODALITY_WORDS = ["может", "должен", "обязан", "вправе", "имеет право"]

# ========== 🧠 ЗАГРУЗКА МОДЕЛЕЙ ==========
print("\n" + "=" * 50)
print("🤖 ЗАГРУЗКА НЕЙРОСЕТЕЙ...")
print("=" * 50)

if os.path.exists(PATH_TO_LABSE):
    print(f"🧠 LaBSE (Offline): {PATH_TO_LABSE}")
    embedder = SentenceTransformer(PATH_TO_LABSE)
else:
    print("🌐 LaBSE (Online): cointegrated/LaBSE-en-ru")
    embedder = SentenceTransformer("cointegrated/LaBSE-en-ru")

if os.path.exists(PATH_TO_LLM):
    print(f"🔧 Qwen: {os.path.basename(PATH_TO_LLM)}")
    llm = Llama(
        model_path=PATH_TO_LLM,
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,
        n_batch=512,
        verbose=False,
        flash_attn=True,
        #split_mode=1,
        #main_gpu=1,
        tensor_split=[0.5, 0.5],
    )
else:
    print(f"❌ LLM не найдена: {PATH_TO_LLM}")
    exit()

print("✅ Системы готовы\n")

# ========== 🏗️ СОСТОЯНИЕ АГЕНТА ==========


def merge_dicts(existing: Dict, new: Dict) -> Dict:
    """
    Редюсер для слияния словарей статистики
    """
    merged = existing.copy()
    merged.update(new)
    return merged


class AgentState(TypedDict):
    doc1_path: str
    doc2_path: str
    chunks_old: List[str]
    chunks_new: List[str]
    matches: List[Dict]
    filtered_matches: List[Dict]
    analysis_results: List[Dict]
    final_report: str
    logs: Annotated[List[str], operator.add]
    stats: Annotated[Dict, merge_dicts]


# ========== 🛠️ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========


def smart_chunk(text: str) -> List[str]:
    """Умная нарезка с сохранением иерархии документа"""
    chunks = []
    # Разделение по разделам: "1. ЗАГОЛОВОК" или "1.1. Подраздел"
    section_pattern = r"\n(?=\d+\.(?:\d+\.)*\s+[А-ЯЁ])"
    sections = re.split(section_pattern, text)

    for section in sections:
        # Большие разделы дополнительно режем по абзацам
        if len(section) > 2000:
            paragraphs = re.split(r"\n\s*\n", section)
            for p in paragraphs:
                clean = " ".join(p.split())
                if len(clean) > 40:
                    chunks.append(clean)
        else:
            clean = " ".join(section.split())
            if len(clean) > 40:
                chunks.append(clean)

    return chunks


def extract_text_from_pdf(path: str) -> str:
    """Извлечение текста с обработкой ошибок"""
    try:
        if not os.path.exists(path):
            print(f"❌ Файл не найден: {path}")
            return ""

        doc = pymupdf.open(path)

        if doc.is_encrypted:
            print(f"❌ Документ защищен паролем: {path}")
            doc.close()
            return ""

        text = "".join([page.get_text() for page in doc])
        doc.close()
        return text

    except Exception as e:
        print(f"❌ Ошибка чтения PDF {path}: {e}")
        return ""


def compute_similarity_batched(emb_old, emb_new, batch_size=100):
    """
    КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Двусторонний батчинг матрицы сходства
    Экономия RAM: вместо загрузки всей матрицы (500x500 = 1GB)
    вычисляем блоками (100x100 = 40MB за итерацию)
    """
    n_old, n_new = len(emb_old), len(emb_new)
    cosine_scores = np.zeros((n_old, n_new), dtype=np.float32)

    # Конвертация в тензоры если нужно
    if not isinstance(emb_old, torch.Tensor):
        emb_old = torch.tensor(emb_old)
    if not isinstance(emb_new, torch.Tensor):
        emb_new = torch.tensor(emb_new)

    # Двойной цикл батчинга
    for i in range(0, n_old, batch_size):
        batch_old = emb_old[i : i + batch_size]

        for j in range(0, n_new, batch_size):
            batch_new = emb_new[j : j + batch_size]

            # Вычисление маленького блока матрицы
            chunk_scores = util.cos_sim(batch_old, batch_new).cpu().numpy()
            cosine_scores[i : i + len(batch_old), j : j + len(batch_new)] = chunk_scores

            # Освобождение памяти
            del chunk_scores
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return cosine_scores


def should_analyze_change(match: Dict) -> bool:
    """
    КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Умный фильтр приоритизации
    Проверяет: ключевые слова, модальность, отрицания, цифры
    """
    match_type = match["type"]

    # Структурные изменения всегда важны
    if match_type in ["ADDED", "DELETED"]:
        return True

    old_text = match.get("old", "").lower()
    new_text = match.get("new", "").lower()
    text_combined = old_text + " " + new_text

    # 1. Критичные ключевые слова
    if any(kw in text_combined for kw in CRITICAL_KEYWORDS):
        return True

    # 2. Наличие цифр (сроки, суммы, проценты)
    if re.search(r"\d+", text_combined):
        return True

    # 3. ВАЖНО: Изменение модальности (может → должен)
    old_modality = {w for w in MODALITY_WORDS if w in old_text}
    new_modality = {w for w in MODALITY_WORDS if w in new_text}
    if old_modality != new_modality:
        return True

    # 4. ВАЖНО: Изменение отрицания
    old_negation = "не " in old_text or "нет " in old_text
    new_negation = "не " in new_text or "нет " in new_text
    if old_negation != new_negation:
        return True

    # 5. Низкое сходство = существенная правка
    if match["score"] < 0.85:
        return True

    return False


def create_qwen_prompt(old_text: str, new_text: str) -> str:
    """ChatML промпт для Qwen 2.5"""
    return f"""<|im_start|>system
Ты - юрист-эксперт по анализу нормативных документов РФ. Твоя задача - находить юридически значимые изменения.<|im_end|>
<|im_start|>user
Сравни две версии пункта нормативного документа.

СТАРАЯ ВЕРСИЯ:
{old_text}

НОВАЯ ВЕРСИЯ:
{new_text}

Найди ТОЛЬКО юридически значимые изменения:
- Изменение сроков (дней, месяцев, лет)
- Изменение прав/обязанностей ("вправе" → "обязан", "может" → "должен")
- Изменение размеров штрафов, выплат, процентов
- Изменение пунктуации, влияющей на смысл
- Добавление/удаление отрицания ("не", "нет")
- Добавление/удаление условий

ИГНОРИРУЙ:
- Перестановку слов без изменения смысла
- Замену синонимов ("организация" → "учреждение")
- Исправление опечаток

Ответ СТРОГО в JSON формате:
{{
  "is_critical": true/false,
  "diff": "краткое описание изменения",
  "legal_impact": "юридические последствия"
}}
<|im_end|>
<|im_start|>assistant
"""


def parse_llm_response(response: str) -> Dict:
    """Надежный парсинг JSON с валидацией и fallback"""
    try:
        # Убираем markdown
        cleaned = re.sub(r"```json|```", "", response).strip()

        # Поиск JSON в тексте
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)

        if json_match:
            data = json.loads(json_match.group())

            # Валидация обязательных полей
            if "is_critical" not in data:
                data["is_critical"] = True
            if "diff" not in data:
                data["diff"] = "Описание отсутствует"
            if "legal_impact" not in data:
                data["legal_impact"] = "Требуется дополнительный анализ"

            return data
        else:
            # JSON не найден - используем текст как описание
            return {
                "is_critical": True,
                "diff": response[:300],
                "legal_impact": "Ошибка структурирования ответа",
            }

    except json.JSONDecodeError as e:
        print(f"   ⚠️ JSON парсинг: {str(e)[:50]}")
        return {
            "is_critical": True,
            "diff": response[:300],
            "legal_impact": "Ошибка парсинга JSON",
        }
    except Exception as e:
        print(f"   ⚠️ Неожиданная ошибка: {str(e)[:50]}")
        return {
            "is_critical": False,
            "diff": "Критическая ошибка обработки",
            "legal_impact": "Требуется ручная проверка",
        }


# ========== 🛠️ УЗЛЫ ГРАФА ==========


def load_and_chunk_node(state: AgentState):
    """Узел 1: Загрузка PDF и умная нарезка"""
    print("📂 [1/5] Загрузка и нарезка документов...")

    text1 = extract_text_from_pdf(state["doc1_path"])
    text2 = extract_text_from_pdf(state["doc2_path"])

    if not text1 or not text2:
        print("❌ Не удалось извлечь текст")
        return {
            "chunks_old": [],
            "chunks_new": [],
            "logs": ["Ошибка извлечения текста"],
            "stats": {"chunks_old": 0, "chunks_new": 0},
        }

    c1 = smart_chunk(text1)
    c2 = smart_chunk(text2)

    log = f"Абзацев: Doc1={len(c1)}, Doc2={len(c2)}"
    print(f"   ✅ {log}")

    return {
        "chunks_old": c1,
        "chunks_new": c2,
        "logs": [log],
        "stats": {"chunks_old": len(c1), "chunks_new": len(c2)},
    }


def vector_matching_node(state: AgentState):
    """Узел 2: Оптимальное семантическое сопоставление"""
    print("🧠 [2/5] Семантическое сравнение (LaBSE + Hungarian)...")

    c_old = state["chunks_old"]
    c_new = state["chunks_new"]

    if not c_old or not c_new:
        return {"matches": [], "logs": ["Нет данных"], "stats": {}}

    # Векторизация
    print("   Векторизация...")
    emb_old = embedder.encode(c_old, show_progress_bar=False)
    emb_new = embedder.encode(c_new, show_progress_bar=False)

    # Батчинг матрицы (FIX: память)
    print("   Матрица сходства (с батчингом)...")
    if len(c_old) * len(c_new) > 100000:
        cosine_scores = compute_similarity_batched(
            emb_old, emb_new, BATCH_SIZE_SIMILARITY
        )
    else:
        cosine_scores = (
            util.cos_sim(torch.tensor(emb_old), torch.tensor(emb_new)).cpu().numpy()
        )

    # Венгерский алгоритм (FIX: коллизии)
    print("   Оптимизация пар (Hungarian)...")
    cost_matrix = 1 - cosine_scores
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    matched_new = set()

    for r, c in zip(row_ind, col_ind):
        score = float(cosine_scores[r, c])
        txt_old = c_old[r]
        txt_new = c_new[c]

        if score >= THRESHOLD_IDENTICAL:
            matched_new.add(c)  # Идентичные - пропускаем
        elif score >= THRESHOLD_MODIFIED:
            matches.append(
                {"type": "MODIFIED", "score": score, "old": txt_old, "new": txt_new}
            )
            matched_new.add(c)
        else:
            matches.append({"type": "DELETED", "score": 0.0, "old": txt_old, "new": ""})

    # Поиск добавленных блоков
    for j, txt in enumerate(c_new):
        if j not in matched_new:
            matches.append({"type": "ADDED", "score": 0.0, "old": "", "new": txt})

    log = f"Отличий: {len(matches)}"
    print(f"   ✅ {log}")

    return {"matches": matches, "logs": [log], "stats": {"total_matches": len(matches)}}


def prioritize_node(state: AgentState):
    """Узел 3: Интеллектуальная приоритизация"""
    print("🔍 [3/5] Приоритизация изменений...")

    matches = state["matches"]
    filtered = [m for m in matches if should_analyze_change(m)]
    skipped = len(matches) - len(filtered)

    log = f"Фильтр: {len(filtered)} важных (пропущено шума: {skipped})"
    print(f"   ✅ {log}")

    return {
        "filtered_matches": filtered,
        "logs": [log],
        "stats": {"filtered": len(filtered), "skipped": skipped},
    }


def llm_analysis_node(state: AgentState):
    """Узел 4: LLM анализ с улучшенным промптом"""
    print("🤖 [4/5] LLM Анализ юридического смысла...")

    matches = state["filtered_matches"]

    if not matches:
        print("   ℹ️ Нет изменений для анализа")
        return {"analysis_results": []}

    results = []
    total = len(matches)

    for i, item in enumerate(matches):
        print(f"   Обработка {i+1}/{total} [{item['type']}]...", end="\r")

        try:
            if item["type"] == "MODIFIED":
                prompt = create_qwen_prompt(item["old"], item["new"])
                output = llm(prompt, max_tokens=400, temperature=0.1, echo=False)
                response = output["choices"][0]["text"].strip()

                data = parse_llm_response(response)

                # Фильтруем только значимые изменения
                if data.get("is_critical") or len(data.get("diff", "")) > 10:
                    results.append(
                        {
                            "type": "MODIFIED",
                            "critical": data.get("is_critical", False),
                            "diff": data.get("diff", ""),
                            "impact": data.get("legal_impact", ""),
                            "old": item["old"],
                            "new": item["new"],
                            "score": item["score"],
                        }
                    )

            elif item["type"] == "ADDED":
                # Краткий анализ добавлений
                text_preview = item["new"][:1000]
                prompt = f"""<|im_start|>system
Ты - юрист-эксперт.<|im_end|>
<|im_start|>user
Проанализируй новый текст:
{text_preview}

Кратко (1-2 предложения): какие новые обязательства/права это вводит?<|im_end|>
<|im_start|>assistant
"""
                output = llm(prompt, max_tokens=200, temperature=0.2, echo=False)
                response = output["choices"][0]["text"].strip()

                results.append(
                    {
                        "type": "ADDED",
                        "critical": False,
                        "diff": (
                            response if len(response) > 10 else "Добавлен новый раздел"
                        ),
                        "impact": "Расширение правового регулирования",
                        "new": item["new"],
                    }
                )

            elif item["type"] == "DELETED":
                results.append(
                    {
                        "type": "DELETED",
                        "critical": True,
                        "diff": "Раздел полностью удален",
                        "impact": "Сужение правового регулирования",
                        "old": item["old"],
                    }
                )

        except Exception as e:
            print(f"\n   ⚠️ Ошибка анализа: {str(e)[:50]}")

    print(f"\n   ✅ Результатов: {len(results)}")

    return {"analysis_results": results, "logs": [f"Проанализировано: {len(results)}"]}


def report_generation_node(state: AgentState):
    """Узел 5: Структурированный отчет"""
    print("📝 [5/5] Генерация отчета...")

    results = state["analysis_results"]

    # Группировка по критичности
    critical = [r for r in results if r.get("critical")]
    non_critical = [
        r for r in results if not r.get("critical") and r["type"] == "MODIFIED"
    ]
    added = [r for r in results if r["type"] == "ADDED"]
    deleted = [r for r in results if r["type"] == "DELETED"]

    doc1_name = os.path.basename(state["doc1_path"])
    doc2_name = os.path.basename(state["doc2_path"])

    # Заголовок с метаданными
    report = f"""# ⚖️ Отчет о сравнении нормативных документов

**Дата анализа:** {time.strftime('%Y-%m-%d %H:%M:%S')}

## 📄 Документы
- **Старая версия:** `{doc1_name}`
- **Новая версия:** `{doc2_name}`

## 📊 Сводная статистика

| Категория | Количество |
|-----------|------------|
| 🔴 **Критичные изменения** | **{len(critical)}** |
| 🟡 Уточнения формулировок | {len(non_critical)} |
| ✅ Добавлено разделов | {len(added)} |
| ❌ Удалено разделов | {len(deleted)} |
| **ИТОГО изменений** | **{len(results)}** |

### Детали обработки
- Абзацев в старой версии: {state.get('stats', {}).get('chunks_old', 'N/A')}
- Абзацев в новой версии: {state.get('stats', {}).get('chunks_new', 'N/A')}
- Обнаружено отличий: {state.get('stats', {}).get('total_matches', 'N/A')}
- Отфильтровано для анализа: {state.get('stats', {}).get('filtered', 'N/A')}

---

"""

    # Критичные изменения (в начале!)
    if critical:
        report += "## 🔴 КРИТИЧНЫЕ ИЗМЕНЕНИЯ (требуют немедленного внимания)\n\n"
        for i, item in enumerate(critical, 1):
            sim = int(item.get("score", 0) * 100) if item.get("score") else "N/A"
            report += f"### {i}. Изменение условий"
            if sim != "N/A":
                report += f" (сходство: {sim}%)"
            report += "\n\n"
            report += f"**Суть изменения:** {item.get('diff', 'Не указано')}\n\n"
            report += f"**Юридические последствия:** {item.get('impact', 'Требует анализа')}\n\n"

            if "old" in item:
                report += f"🔴 **Было:**\n> {item['old'][:400]}{'...' if len(item['old']) > 400 else ''}\n\n"
            if "new" in item:
                report += f"🟢 **Стало:**\n> {item['new'][:400]}{'...' if len(item['new']) > 400 else ''}\n\n"

            report += "---\n\n"

    # Некритичные изменения
    if non_critical:
        report += "## 🟡 Некритичные изменения (уточнения формулировок)\n\n"
        for item in non_critical:
            report += f"- {item.get('diff', 'Изменение формулировки')}\n"
        report += "\n---\n\n"

    # Добавленные разделы
    if added:
        report += "## ✅ Добавленные разделы\n\n"
        for item in added:
            report += f"**Суть:** {item.get('diff', 'Новый раздел')}\n"
            report += (
                f"> {item['new'][:300]}{'...' if len(item['new']) > 300 else ''}\n\n"
            )
        report += "---\n\n"

    # Удаленные разделы
    if deleted:
        report += "## ❌ Удаленные разделы\n\n"
        for item in deleted:
            report += (
                f"> {item['old'][:300]}{'...' if len(item['old']) > 300 else ''}\n\n"
            )

    # Футер
    report += """
---

*Отчет сгенерирован автоматически агентной системой на базе:*
- *LangGraph (оркестрация)*
- *Qwen 2.5 14B (юридический анализ)*
- *LaBSE (семантическое сопоставление)*
"""

    if not results:
        report += (
            "\n\n## ✅ Заключение\n\nДокументы **идентичны** по юридическому смыслу."
        )

    print(f"   ✅ Отчет готов ({len(report)} символов)")

    return {"final_report": report}


# ========== 🕸️ СБОРКА ГРАФА ==========


def build_agent():
    """Построение агентного графа"""
    workflow = StateGraph(AgentState)

    workflow.add_node("load", load_and_chunk_node)
    workflow.add_node("match", vector_matching_node)
    workflow.add_node("prioritize", prioritize_node)
    workflow.add_node("analyze", llm_analysis_node)
    workflow.add_node("report", report_generation_node)

    workflow.set_entry_point("load")
    workflow.add_edge("load", "match")
    workflow.add_edge("match", "prioritize")
    workflow.add_edge("prioritize", "analyze")
    workflow.add_edge("analyze", "report")
    workflow.add_edge("report", END)

    return workflow.compile()


# ========== 🚀 ЗАПУСК ==========

if __name__ == "__main__":
    agent_app = build_agent()

    doc1 = r"/home/seral/diploma/dev_1_conda/documents/H12100110_1621890000.pdf"
    doc2 = r"/home/seral/diploma/dev_1_conda/documents/H12300274_1688590800.pdf"

    print("\n" + "=" * 60)
    print("🚀 ЗАПУСК АГЕНТНОЙ СИСТЕМЫ СРАВНЕНИЯ ДОКУМЕНТОВ")
    print("=" * 60 + "\n")

    if not os.path.exists(doc1):
        print(f"❌ Файл не найден: {doc1}")
        exit()

    if not os.path.exists(doc2):
        print(f"❌ Файл не найден: {doc2}")
        exit()

    try:
        start_time = time.time()

        final_state = agent_app.invoke(
            {
                "doc1_path": doc1,
                "doc2_path": doc2,
                "chunks_old": [],
                "chunks_new": [],
                "matches": [],
                "filtered_matches": [],
                "analysis_results": [],
                "final_report": "",
                "logs": [],
                "stats": {},
            }
        )

        elapsed = time.time() - start_time

        # Сохранение отчета
        report_file = f"Report_{time.strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(final_state["final_report"])

        # Сохранение логов
        log_file = f"Logs_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "logs": final_state.get("logs", []),
                    "stats": final_state.get("stats", {}),
                    "elapsed_seconds": round(elapsed, 2),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print("\n" + "=" * 60)
        print("🎉 АНАЛИЗ ЗАВЕРШЕН")
        print("=" * 60)
        print(f"⏱️  Время: {elapsed:.1f} сек")
        print(f"📄 Отчет: {report_file}")
        print(f"📋 Логи: {log_file}\n")

    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback

        traceback.print_exc()

    finally:
        if 'llm' in globals() and llm is not None:
            try:
                llm.close() # Попытка корректного закрытия
            except:
                pass
            # del llm # Не удаляем объект вручную, чтобы не триггерить багованный __del__
            print("🧹 Работа завершена")
