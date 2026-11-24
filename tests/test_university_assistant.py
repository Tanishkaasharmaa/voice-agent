from university.assistant import UniversityAssistant, detect_mode_command
from university.knowledge_base import load_faqs
from university.retriever import FAQRetriever


def test_loads_all_eighteen_wifi_faqs():
    assert len(load_faqs()) == 18


def test_retrieves_samarth_application():
    assistant = UniversityAssistant()
    answer = assistant.answer("How can I apply for Wi-Fi through Samarth?")
    assert "Samarth portal" in answer


def test_retrieves_windows_mac_instructions():
    assistant = UniversityAssistant()
    answer = assistant.answer("How do I find my MAC address on Windows?")
    assert "ipconfig /all" in answer


def test_does_not_invent_answer_for_unmatched_question():
    assistant = UniversityAssistant()
    answer = assistant.answer("What is the university library closing time?")
    assert "couldn't find" in answer.lower()


def test_detects_explicit_mode_switches_only():
    assert detect_mode_command("switch to university help desk") == "university"
    assert detect_mode_command("switch to general assistant") == "general"
    assert detect_mode_command("How do I switch my Wi-Fi device?") is None


def test_retriever_returns_no_match_for_empty_input():
    retriever = FAQRetriever(load_faqs())
    assert retriever.search("please") == (None, 0.0, 0.0)


def test_does_not_match_unrelated_university_question():
    assistant = UniversityAssistant()
    answer = assistant.answer("What is the university library closing time?")
    assert "couldn't find" in answer.lower()
