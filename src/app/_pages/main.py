import src.app._pages.home as home
import src.app._pages.sentiment_analysis.main as sentiment_analysis
import src.app._pages.topic_modelling.main as topic_modelling
import src.app.navigation as navigation

router = {
    "Home": home,
    "Sentiment Analysis": sentiment_analysis,
    "Topic Modelling": topic_modelling,
}


def display():
    navigation.sidebar_router(router, label="Spaces", level=1)
