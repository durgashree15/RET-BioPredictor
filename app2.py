import os
os.system("pip install -r requirements.txt")
import streamlit as st
import pandas as pd
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors
import joblib

def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is not None:
        # Calculate all available descriptors
        descriptor_names = [desc[0] for desc in Descriptors._descList]
        descriptor_values = [Descriptors.MolWt(mol)]  # Initialize with molecular weight

        for desc_name in descriptor_names[1:]:  # Skip molecular weight (already added)
            descriptor_values.append(getattr(Descriptors, desc_name)(mol))

        # Return a dictionary of descriptors
        descriptors = dict(zip(descriptor_names, descriptor_values))
        return descriptors
    else:
        print(f"Failed to parse SMILES: {smiles}")
        return None
    
def process_data(df):
    l=[]
    for i in range(df.shape[0]):
        if (df.iloc[i]["descriptors"]) is None:
            l.append(i)
            print(df.loc[i,0])
    
    df=df.drop(l, axis=0)

    df_descriptors = pd.DataFrame(df['descriptors'].tolist(), index=df.index)

    # Concatenate the original DataFrame with the descriptors DataFrame
    df = pd.concat([df, df_descriptors], axis=1)

    # Drop the 'descriptors' column if you don't need it anymore
    df = df.drop('descriptors', axis=1)

    model=joblib.load("gb_imp.pkl")

    Bioactivity = model.predict(df.drop("Ligand SMILES", axis=1))

    result = pd.DataFrame({
        'Column1': df["Ligand SMILES"],
        'Column2': Bioactivity
    })
    csv = result.to_csv(index=False)
    return csv

st.set_page_config(layout="wide")
# Title and tagline

st.markdown(
    """
    <h1 style="text-align: center; background-color: #2196F3; padding: 20px; color: white;border-radius:10px">
        RET-BioPredictor
    </h1>
    """,
    unsafe_allow_html=True
)

# Center-align the subheader
st.markdown(
    """
    <h2 style="text-align: center; color: #FFD54F;">
        Predict bioactivity data of compounds against RET V804M Mutation
    </h2>
    """,
    unsafe_allow_html=True
)

# Divide the page into two sections
left, right = st.columns([2, 1])

# Left section - problem description, image, and instructions
with left:
    st.markdown(
        """
        <p style="font-size: 23px;">
            RET (rearranged during  transfection) mutations are frequently associated with hereditary forms of Medullary Thyroid Carcinoma (MTC) and Familial Medullary Thyroid Carcinoma (FMTC) (Takahashi, Masahide et al., 2020).The RETV804M also known as a \"gatekeeper\" mutation is well-known for conferring resistance to certain Tyrosine Kinase Inhibitors (TKIs). New-generation RET inhibitors, such as 'Selpercatinib' and 'Pralsetinib', have been developed to target RET mutations, including V804M (Subbiah, V et al., 2021) (Subbiah, Vivek, and Gilbert J. Cote., 2020)We have utilized biological activity data(IC50) from all compounds tested against RET V804M mutation from BindingDB, and using RDKit, we extracted descriptors to develop a machine learning model for predicting bioactivity.
        </p>
        """,
        unsafe_allow_html=True
    )
    
    # Add an image (replace 'your_image.png' with the path to your image)

    # img = Image.open("ret.jpg")  # Replace with your image path
    # img = img.resize((300, 300))  # Resize to 300x300 pixels
    left_inner, middle_inner, right_inner = st.columns([1, 2, 1])
    with middle_inner:
        st.image("ret.jpg", width=500, caption="Figure1. Cartoon representation of the RET protein (PDB ID 7JU6) overlaid  with a molecular surface representation in blue, showing drug Selpercatinib bound in space-filling model colored yellow.")
    
        

    st.markdown(
            """
            <h4 style="color: white">
            References
            </h4>
            <div style="font-size: 25px;">
                <ol>
                    <li>Takahashi, Masahide et al. “Roles of the RET Proto-oncogene in Cancer and Development.” JMA journal vol. 3,3 (2020): 175-181. doi:10.31662/jmaj.2020-0021</li>
                    <li>Subbiah, V et al. “Structural basis of acquired resistance to selpercatinib and pralsetinib mediated by non-gatekeeper RET mutations.” Annals of oncology : official journal of the European Society for Medical Oncology vol. 32,2 (2021): 261-268. doi:10.1016/j.annonc.2020.10.599</li>
                    <li>Subbiah, Vivek, and Gilbert J. Cote. "Advances in targeting RET-dependent cancers." Cancer discovery 10.4 (2020): 498-505</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True
        )

# Right section - Text input and file upload
with right:
    st.markdown(
    "<h3 style='font-size:24px;'>Enter comma-separated SMILES sequences</h3>",
    unsafe_allow_html=True
    )
    # Text bar for user input
    user_text = st.text_input("Paste here:")
    if user_text:
        u = user_text.split(",")
    else:
        u = [] 
    
    # File uploader for CSV files
    st.markdown(
    "<h3 style='font-size:24px;'>Choose a CSV file</h3>",
    unsafe_allow_html=True
    )
    uploaded_file = st.file_uploader("Browse here", type=["csv","txt"])

    if st.button("Predict Biological Activity"):
        received = 0
        if len(u)!=0:
            st.write("SMILES strings submitted successfully!")
            df = pd.DataFrame(u, columns=['Ligand SMILES'])
            received = 1 

            df['descriptors'] = df["Ligand SMILES"].apply(calculate_descriptors)

        elif uploaded_file is not None:
            st.write("File submitted successfully!")
            # Check the file type and read accordingly
            if uploaded_file.type == "csv":
                df = pd.read_csv(uploaded_file, header=0)       
            else:
                df = pd.read_csv(uploaded_file, delimiter=',')
            received = 1
            df['descriptors'] = df["Ligand SMILES"].apply(calculate_descriptors)
        else:
            st.warning("Please paste SMILES strings or upload a CSV or TXT file before submitting.")
    
    
        if received == 1:
            with st.spinner("Processing..."):
                csv = process_data(df)
            
            st.success("Processing completed! Your results are ready for download.")

            # Display the download button after processing
            st.download_button(
                label="Download Results",
                data=csv,
                file_name="processed_results.csv",
                mime="text/csv"
            )

    st.markdown(
            """
            <h4 style="color: blue">
            Instructions for Use:
            </h4>
            <div style="font-size: 25px;">
                <ul>
                    <li>To test the model for a few molecules, you can enter their SMILES strings, separated by commas, directly in the search bar.</li>
                    <li>To test a large number of molecules, upload a file with 'Ligand SMILES' as column title for CSV files or 'Ligand SMILES' as header with comma-separated SMILES strings for TXT files.</li>
                    <li>Click on 'Predict Biological Activity' button to download the CSV file with predictions.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <h4 style="color: blue">
        Note:
        </h4>
        <p style="color: white; font-size=30px">
        1. Wait for the download button to pop-up to download the CSV file with results.
        </p>
        <p style="color: white; font-size=25px">
        2. Predictions tend to be more reliable when the structure of the compounds being predicted closely resemble those in the training set.
        </p>
        """,
    unsafe_allow_html=True
    )
    

        
