# app.py
import streamlit as st
from model_classes import BackgroundRemover, ImageCache
from ui_components import show_batch_processor_tab, show_comparison_tab, init_session_state, show_settings

def main():
    st.title("Background Remover Demo")
    
    if 'model' not in st.session_state:
        st.session_state.model = BackgroundRemover()
    if 'image_cache' not in st.session_state:
        st.session_state.image_cache = ImageCache()
        
    init_session_state()
    
    with st.sidebar:
        show_settings()  # Только этот вызов
    
    tab1, tab2 = st.tabs(["Batch Processor", "Model Comparison"])
    
    with tab1:
        show_batch_processor_tab()
    
    with tab2:
        show_comparison_tab()
if __name__ == "__main__":
    main()