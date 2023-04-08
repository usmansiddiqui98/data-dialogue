import src.app._pages.topic_modelling.bert_topic as bert_topic
import src.app.navigation as navigation

router = {
    "BERTopic": bert_topic,
}


def display():
    navigation.sidebar_router(router=router, label="Pages", level=2)
