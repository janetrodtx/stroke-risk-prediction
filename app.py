# Determine risk category
if risk_score < 0.3:
    st.success("You are at **Low Risk** for stroke. Keep up the healthy habits!")
    st.write("✅ **Recommendations:**")
    st.write("- Maintain regular physical activity (150 mins/week).")
    st.write("- Follow a balanced diet (low sodium, rich in fruits and vegetables).")
    st.write("- Keep regular health checkups for blood pressure and cholesterol.")
elif risk_score < 0.6:
    st.warning("You are at **Medium Risk** for stroke. Consider making lifestyle improvements.")
    st.write("⚠️ **Recommendations:**")
    st.write("- Increase physical activity: Aim for 30 minutes of exercise, 5 days a week.")
    st.write("- Quit smoking: Seek support or resources.")
    st.write("- Monitor cholesterol and blood pressure regularly.")
else:
    st.error("You are at **High Risk** for stroke. Take immediate action.")
    st.write("🚨 **Recommendations:**")
    st.write("- Consult a healthcare provider immediately.")
    st.write("- Implement lifestyle changes: Quit smoking, reduce salt intake.")
    st.write("- Increase physical activity to manage weight and cardiovascular health.")
    st.write("- Consider medication for blood pressure and cholesterol if advised.")

# Footer
st.markdown("---")
st.markdown("📋 **Note:** This prediction is based on the data provided and is not a substitute for professional medical advice.")


