import streamlit as st


def sidebar_router(router, label: str = "", level: int = 1):
    # Get options
    options = list(router.keys())

    # Get query params
    param_key = f"page{level}"

    # Display options
    try:
        index = options.index(param_key)
    except Exception:
        index = 0
    option = st.sidebar.selectbox(label, options, index=index, key=param_key)
    st.sidebar.markdown("---")

    # Display selection
    router[option].display()
