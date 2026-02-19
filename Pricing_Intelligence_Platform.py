import streamlit as st

st.set_page_config(page_title="Pricing Analytics Portfolio", layout="wide")

st.title("Pricing Analytics Portfolio")
st.caption("Interactive POCs for Distribution Pricing (Synthetic data)")

st.markdown("""
### How to navigate
Use the **left sidebar** to open:
- **Price Raise Engine:** Elasticity + scenario simulation + raise score + action table  
- **Discount Leakage Engine:** Customer segmentation + leakage detection + rep leaderboard  

### Notes
- All data is **synthetic** and for demonstration only.
- These POCs are designed to show pricing decision workflows: raise opportunities, leakage control, and governance.
""")
