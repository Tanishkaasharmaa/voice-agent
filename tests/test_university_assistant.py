from university.assistant import (
    OUT_OF_SCOPE_RESPONSE,
    REPHRASE_RESPONSE,
    UniversityAssistant,
    detect_mode_command,
)
from university.knowledge_base import load_faqs
from university.retriever import FAQRetriever
from university.scope_classifier import (
    OUT_OF_SCOPE,
    UNIVERSITY_UNCLEAR,
    classify_scope,
)


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


def test_university_support_question_without_faq_match_asks_to_rephrase():
    assistant = UniversityAssistant()
    answer = assistant.answer("I have a problem with my student account")
    assert answer == REPHRASE_RESPONSE


def test_paraphrased_network_issue_asks_to_rephrase():
    assistant = UniversityAssistant()
    answer = assistant.answer("The internet keeps dropping in my hostel room")
    assert answer == REPHRASE_RESPONSE


def test_general_knowledge_question_gets_out_of_scope_response():
    assistant = UniversityAssistant()
    answer = assistant.answer("What is the capital of India?")
    assert answer == OUT_OF_SCOPE_RESPONSE


def test_university_trivia_is_not_automatically_helpdesk_scope():
    assistant = UniversityAssistant()
    answer = assistant.answer("What time does the university library close?")
    assert answer == OUT_OF_SCOPE_RESPONSE


def test_scope_classifier_separates_three_outcomes_after_faq_lookup():
    assert classify_scope("my internet keeps disconnecting") == UNIVERSITY_UNCLEAR
    assert classify_scope("I need to update my hostel address") == UNIVERSITY_UNCLEAR
    assert classify_scope("What is the capital of India?") == OUT_OF_SCOPE
    assert classify_scope("Tell me about university history") == OUT_OF_SCOPE


def test_detects_explicit_mode_switches_only():
    assert detect_mode_command("switch to university help desk") == "university"
    assert detect_mode_command("switch to general assistant") == "general"
    assert detect_mode_command("How do I switch my Wi-Fi device?") is None


def test_retriever_returns_no_match_for_empty_input():
    retriever = FAQRetriever(load_faqs())
    assert retriever.search("please") == (None, 0.0, 0.0)
