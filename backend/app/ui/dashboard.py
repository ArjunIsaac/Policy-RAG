import streamlit as st
import requests

API_URL = "http://localhost:8000"


st.set_page_config(page_title="Insurance RAG System", layout="wide")

st.title("📄 Insurance Policy Intelligence System")

# -------------------------
# UPLOAD SECTION
# -------------------------
st.header("1. Upload Policy PDF")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:

    if st.button("Process PDF"):

        files = {"file": uploaded_file}

        response = requests.post(
            f"{API_URL}/upload",
            files={"file": uploaded_file}
        )

        st.success(response.json())


# -------------------------
# SINGLE ATTRIBUTE EXTRACTION
# -------------------------
st.header("2. Extract Attribute")

attribute = st.selectbox(
    "Select Attribute",
    [
        "insurer",
        "policy_name",
        "free_look_period",
        "grace_period",
        "ped_waiting_period",
        "specific_disease_waiting_period",
        "initial_waiting_period",
        "room_rent_limit",
        "icu_limit",
        "copay",
        "cashless_hospitals",
        "portability"
    ]
)

if st.button("Extract"):

    response = requests.get(
        f"{API_URL}/extract/{attribute}"
    )

    data = response.json()

    st.subheader("Result")

    st.json(data)

    # -------------------------
    # VISUAL HIGHLIGHTS
    # -------------------------
    st.subheader("Value")
    st.write(data.get("value"))

    st.subheader("Confidence")
    st.write(data.get("confidence"))

    st.subheader("Evidence")
    st.write(data.get("evidence"))

    st.subheader("Page")
    st.write(data.get("page"))

    # -------------------------
    # CONFLICTS
    # -------------------------
    st.subheader("Conflicts")

    conflicts = data.get("conflicts", [])

    if conflicts:
        for c in conflicts:
            st.warning(f"Page {c['page']}: {c['text']}")
    else:
        st.success("No conflicts detected")


# -------------------------
# FULL POLICY EXTRACTION
# -------------------------
st.header("3. Extract Full Policy")

if st.button("Run Full Extraction"):

    response = requests.get(f"{API_URL}/extract/all")

    data = response.json()

    st.success("Extraction complete")

    st.json(data)