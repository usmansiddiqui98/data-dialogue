import src.app._pages.topic_modelling.evaluation as evaluation
import src.app.navigation as navigation

router = {"Evaluation": evaluation}


def display():
    navigation.sidebar_router(router=router, label="Pages", level=2)
