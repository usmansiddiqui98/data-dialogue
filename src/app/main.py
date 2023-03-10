import streamlit as st

import src.app._pages.main as pages


def main():
    # sets page layout
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Data Dialogue",
        menu_items={"About": "Data Dialogue"},
    )

    # displays application _pages
    pages.display()


if __name__ == "__main__":
    main()
