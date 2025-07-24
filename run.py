import streamlit as st
from agents_demo.workflow import Workflow  # 导入你的Agent逻辑

st.title("华娴的agents demo")

if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.workflow = Workflow()
#
#
#
# # 用户输入

user_input = st.text_input("请输入您的问题:")
# #
if st.button("运行Agent"):
    with st.spinner("Agent 处理中..."):
        result = st.session_state.workflow.run(user_input)  # 调用你的Agent
        st.success("完成！")
        st.json(result[-1].content)  # 显示结构化结果
        st.markdown("**详细过程:**")
        st.write(result)  # 展示Agent思考过程