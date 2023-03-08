import src.app._pages.sentiment_analysis.sentiment_model as sentiment_model
import src.app.navigation as navigation

router = {
    "Predict": sentiment_model,
}


def display():
    navigation.sidebar_router(router=router, label="Pages", level=2)
