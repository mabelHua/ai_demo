import streamlit as st
from agents import Agent  # 导入你的Agent逻辑

st.title("LLM Agent 控制面板")

if not Agent.workflow:
    agent = Agent()

# 用户输入
user_input = st.text_input("请输入您的问题:")
#
if st.button("运行Agent"):
    with st.spinner("Agent 处理中..."):
        result = agent.run(user_input if user_input else '')  # 调用你的Agent
        st.success("完成！")
        st.json(result[-1].content)  # 显示结构化结果
        st.markdown("**详细过程:**")
        st.write(result)  # 展示Agent思考过程

