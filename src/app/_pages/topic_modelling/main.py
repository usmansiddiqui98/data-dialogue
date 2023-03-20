import src.app._pages.topic_modelling.human_feedback as human_feedback
import src.app.navigation as navigation

router = {
    "Feedback": human_feedback,
}


def display():
    navigation.sidebar_router(router=router, label="Pages", level=2)
