import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import zipfile
import warnings

# Suppress FutureWarnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Page configuration
st.set_page_config(
    page_title="Support Dyson Monthly",
    page_icon="🧮",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e40af 0%, #4f46e5 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
    }
    .main-header p {
        color: #dbeafe;
        margin: 0.5rem 0 0 0;
    }
    .metric-card {
        background: #f0f9ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2563eb;
    }
    .success-box {
        background: #dcfce7;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #16a34a;
    }
    .info-box {
        background: #dbeafe;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2563eb;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        background-color: #f3f4f6;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🧮 Support Dyson Monthly</h1>
    <p>Calculate promotional support for B2B & B2C channels</p>
</div>
""", unsafe_allow_html=True)


def process_b2b_data(b2b_zip, pm_file, dyson_promo_file):
    """Process B2B data and calculate support"""
    try:
        # Read B2B report from ZIP
        with zipfile.ZipFile(b2b_zip) as z:
            csv_name = [name for name in z.namelist() if name.endswith('.csv')][0]
            with z.open(csv_name) as f:
                b2b = pd.read_csv(f)
        
        # Read PM file and Dyson Promo file
        PM = pd.read_excel(pm_file)
        Dyson_Promo = pd.read_excel(dyson_promo_file)
        
        # Clean and prepare data
        b2b["Asin"] = b2b["Asin"].astype(str).str.strip()
        PM["ASIN"] = PM["ASIN"].astype(str).str.strip()
        Dyson_Promo["ASIN"] = Dyson_Promo["ASIN"].astype(str).str.strip()
        
        # Map Brand from PM file
        brand_map = PM.groupby("ASIN", as_index=True)["Brand"].first()
        b2b["Brand"] = b2b["Asin"].map(brand_map)
        
        # Move Brand column after Sku if Sku exists
        cols = list(b2b.columns)
        if "Sku" in cols:
            sku_idx = cols.index("Sku")
            cols.remove("Brand")
            cols.insert(sku_idx + 1, "Brand")
            b2b = b2b[cols]
        
        # Filter Dyson brand - handle NaN values
        dyson_b2b = b2b[b2b["Brand"].notna() & (b2b["Brand"].astype(str).str.strip().str.upper() == "DYSON")].copy()
        
        # Identify cancelled orders
        cancel_data = dyson_b2b[
            dyson_b2b["Transaction Type"].astype(str).str.strip().str.upper() == "CANCEL"
        ]
        cancel_order_set = set(cancel_data["Order Id"])
        
        # Add Order Status
        dyson_b2b["Order Status"] = dyson_b2b["Order Id"].apply(
            lambda x: x if x in cancel_order_set else np.nan
        )
        
        # Move Order Status after Order Id
        cols = list(dyson_b2b.columns)
        order_idx = cols.index("Order Id")
        cols.remove("Order Status")
        cols.insert(order_idx + 1, "Order Status")
        dyson_b2b = dyson_b2b[cols]
        
        # Replace NaN with Transaction Type
        dyson_b2b["Order Status"] = np.where(
            dyson_b2b["Order Status"].isna(),
            dyson_b2b["Transaction Type"],
            "Cancel"
        )
        
        # Create pivot table
        pivot_order_status = pd.pivot_table(
            dyson_b2b,
            index="Asin",
            columns="Order Status",
            values="Quantity",
            aggfunc="sum",
            fill_value=0,
            margins=True,
            margins_name="Grand Total"
        )
        
        pivot_order_status_index = pivot_order_status.reset_index()
        
        # Calculate Net Sale
        pivot_order_status_index["Net Sale / Actual Shipment"] = (
            pivot_order_status_index.get("Shipment", 0) -
            pivot_order_status_index.get("Refund", 0)
        )
        
        # Map SKU, SSP, Cons Promo, Margin from Dyson_Promo file
        sku_map = Dyson_Promo.groupby("ASIN", as_index=True)["SKU Code"].first()
        ssp_map = Dyson_Promo.groupby("ASIN", as_index=True)["SSP"].first()
        cons_map = Dyson_Promo.groupby("ASIN", as_index=True)["Cons Promo"].first()
        margin_map = Dyson_Promo.groupby("ASIN", as_index=True)["Margin"].first()
        
        pivot_order_status_index["SKU CODE"] = pivot_order_status_index["Asin"].map(sku_map)
        pivot_order_status_index["SSP"] = pivot_order_status_index["Asin"].map(ssp_map)
        pivot_order_status_index["Cons Promo"] = pivot_order_status_index["Asin"].map(cons_map)
        pivot_order_status_index["Margin %"] = (
            pivot_order_status_index["Asin"].map(margin_map).mul(100)
        )
        
        # Calculate Support
        pivot_order_status_index["Support"] = (
            (pivot_order_status_index["SSP"] - pivot_order_status_index["Cons Promo"])
            * (1 - pivot_order_status_index["Margin %"] / 100)
        )
        
        # Calculate Support as per Net Sale - NO WARNINGS
        pivot_order_status_index["SUPPORT AS PER NET SALE"] = (
            pd.to_numeric(pivot_order_status_index["Support"], errors='coerce').fillna(0)
            * pd.to_numeric(pivot_order_status_index["Net Sale / Actual Shipment"], errors='coerce').fillna(0)
        )
        
        # Fill NaN with 0 except for Grand Total row - NO WARNINGS
        mask = pivot_order_status_index["Asin"] != "Grand Total"
        for col in pivot_order_status_index.columns:
            if col not in ["Asin", "SKU CODE", "Order Status"]:
                pivot_order_status_index.loc[mask, col] = pd.to_numeric(
                    pivot_order_status_index.loc[mask, col], errors='coerce'
                ).fillna(0)
        
        # Recalculate Grand Total
        df_no_gt = pivot_order_status_index[
            pivot_order_status_index["Asin"] != "Grand Total"
        ].copy()
        
        exclude_cols = ["Asin", "SKU CODE"]
        cols_to_sum = [c for c in df_no_gt.columns if c not in exclude_cols]
        
        df_no_gt[cols_to_sum] = df_no_gt[cols_to_sum].apply(pd.to_numeric, errors="coerce")
        
        grand_total = df_no_gt[cols_to_sum].sum().to_frame().T
        grand_total["Asin"] = "Grand Total"
        grand_total["SKU CODE"] = ""
        grand_total = grand_total[pivot_order_status_index.columns]
        
        pivot_order_status_index = pd.concat([df_no_gt, grand_total], ignore_index=True)
        
        # Convert SKU CODE to string to avoid Arrow serialization issues
        pivot_order_status_index["SKU CODE"] = pivot_order_status_index["SKU CODE"].astype(str)
        
        return pivot_order_status_index
        
    except Exception as e:
        st.error(f"Error processing B2B data: {str(e)}")
        return None


def process_b2c_data(b2c_zip, pm_file, dyson_promo_file):
    """Process B2C data and calculate support"""
    try:
        # Read B2C report from ZIP
        with zipfile.ZipFile(b2c_zip) as z:
            csv_name = [name for name in z.namelist() if name.endswith('.csv')][0]
            with z.open(csv_name) as f:
                b2c = pd.read_csv(f)
        
        # Read PM file and Dyson Promo file
        PM = pd.read_excel(pm_file)
        Dyson_Promo = pd.read_excel(dyson_promo_file)
        
        # Clean and prepare data
        b2c["Asin"] = b2c["Asin"].astype(str).str.strip()
        PM["ASIN"] = PM["ASIN"].astype(str).str.strip()
        Dyson_Promo["ASIN"] = Dyson_Promo["ASIN"].astype(str).str.strip()
        
        # Map Brand from PM file
        brand_map = PM.groupby("ASIN", as_index=True)["Brand"].first()
        b2c["Brand"] = b2c["Asin"].map(brand_map)
        
        # Move Brand column after Sku if Sku exists
        cols = list(b2c.columns)
        if "Sku" in cols:
            sku_idx = cols.index("Sku")
            cols.remove("Brand")
            cols.insert(sku_idx + 1, "Brand")
            b2c = b2c[cols]
        
        # Filter Dyson brand - handle NaN values
        dyson_b2c = b2c[b2c["Brand"].notna() & (b2c["Brand"].astype(str).str.strip().str.upper() == "DYSON")].copy()
        
        # Identify cancelled orders
        cancel_data = dyson_b2c[
            dyson_b2c["Transaction Type"].astype(str).str.strip().str.upper() == "CANCEL"
        ]
        cancel_order_set = set(cancel_data["Order Id"])
        
        # Add Order Status
        dyson_b2c["Order Status"] = dyson_b2c["Order Id"].apply(
            lambda x: x if x in cancel_order_set else np.nan
        )
        
        # Move Order Status after Order Id
        cols = list(dyson_b2c.columns)
        order_idx = cols.index("Order Id")
        cols.remove("Order Status")
        cols.insert(order_idx + 1, "Order Status")
        dyson_b2c = dyson_b2c[cols]
        
        # Replace NaN with Transaction Type
        dyson_b2c["Order Status"] = np.where(
            dyson_b2c["Order Status"].isna(),
            dyson_b2c["Transaction Type"],
            "Cancel"
        )
        
        # Create pivot table
        pivot_order_status = pd.pivot_table(
            dyson_b2c,
            index="Asin",
            columns="Order Status",
            values="Quantity",
            aggfunc="sum",
            fill_value=0,
            margins=True,
            margins_name="Grand Total"
        )
        
        pivot_order_status_index = pivot_order_status.reset_index()
        
        # Calculate Net Sale
        pivot_order_status_index["Net Sale / Actual Shipment"] = (
            pivot_order_status_index.get("Shipment", 0) -
            pivot_order_status_index.get("Refund", 0)
        )
        
        # Map SKU, SSP, Cons Promo, Margin from Dyson_Promo file
        sku_map = Dyson_Promo.groupby("ASIN", as_index=True)["SKU Code"].first()
        ssp_map = Dyson_Promo.groupby("ASIN", as_index=True)["SSP"].first()
        cons_map = Dyson_Promo.groupby("ASIN", as_index=True)["Cons Promo"].first()
        margin_map = Dyson_Promo.groupby("ASIN", as_index=True)["Margin"].first()
        
        pivot_order_status_index["SKU CODE"] = pivot_order_status_index["Asin"].map(sku_map)
        pivot_order_status_index["SSP"] = pivot_order_status_index["Asin"].map(ssp_map)
        pivot_order_status_index["Cons Promo"] = pivot_order_status_index["Asin"].map(cons_map)
        pivot_order_status_index["Margin %"] = (
            pivot_order_status_index["Asin"].map(margin_map).mul(100)
        )
        
        # Calculate Support
        pivot_order_status_index["Support"] = (
            (pivot_order_status_index["SSP"] - pivot_order_status_index["Cons Promo"])
            * (1 - pivot_order_status_index["Margin %"] / 100)
        )
        
        # Calculate Support as per Net Sale - NO WARNINGS
        pivot_order_status_index["SUPPORT AS PER NET SALE"] = (
            pd.to_numeric(pivot_order_status_index["Support"], errors='coerce').fillna(0)
            * pd.to_numeric(pivot_order_status_index["Net Sale / Actual Shipment"], errors='coerce').fillna(0)
        )
        
        # Fill NaN with 0 except for Grand Total row - NO WARNINGS
        mask = pivot_order_status_index["Asin"] != "Grand Total"
        for col in pivot_order_status_index.columns:
            if col not in ["Asin", "SKU CODE", "Order Status"]:
                pivot_order_status_index.loc[mask, col] = pd.to_numeric(
                    pivot_order_status_index.loc[mask, col], errors='coerce'
                ).fillna(0)
        
        # Recalculate Grand Total
        df_no_gt = pivot_order_status_index[
            pivot_order_status_index["Asin"] != "Grand Total"
        ].copy()
        
        exclude_cols = ["Asin", "SKU CODE"]
        cols_to_sum = [c for c in df_no_gt.columns if c not in exclude_cols]
        
        df_no_gt[cols_to_sum] = df_no_gt[cols_to_sum].apply(pd.to_numeric, errors="coerce")
        
        grand_total = df_no_gt[cols_to_sum].sum().to_frame().T
        grand_total["Asin"] = "Grand Total"
        grand_total["SKU CODE"] = ""
        grand_total = grand_total[pivot_order_status_index.columns]
        
        pivot_order_status_index = pd.concat([df_no_gt, grand_total], ignore_index=True)
        
        # Convert SKU CODE to string to avoid Arrow serialization issues
        pivot_order_status_index["SKU CODE"] = pivot_order_status_index["SKU CODE"].astype(str)
        
        return pivot_order_status_index
        
    except Exception as e:
        st.error(f"Error processing B2C data: {str(e)}")
        return None


def convert_df_to_csv(df):
    """Convert dataframe to CSV for download"""
    return df.to_csv(index=False).encode('utf-8')


def format_currency(value):
    """Format number as Indian currency"""
    if pd.isna(value):
        return "-"
    return f"₹{value:,.0f}"


# Main App
tab1, tab2 = st.tabs(["📊 B2B Analysis", "📈 B2C Analysis"])

# B2B Tab
with tab1:
    st.markdown("### B2B Support Calculation")
    
    st.info("📁 Please upload all three required files below:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**1️⃣ B2B Report**")
        b2b_zip = st.file_uploader(
            "Choose B2B ZIP file",
            type=['zip'],
            key='b2b_zip',
            help="Upload b2bReport_October_2025.zip"
        )
        if b2b_zip:
            st.success(f"✅ {b2b_zip.name}")
    
    with col2:
        st.markdown("**2️⃣ PM File**")
        b2b_pm = st.file_uploader(
            "Choose PM Excel file",
            type=['xlsx', 'xls'],
            key='b2b_pm',
            help="Upload PM.xlsx file"
        )
        if b2b_pm:
            st.success(f"✅ {b2b_pm.name}")
    
    with col3:
        st.markdown("**3️⃣ Dyson Promo**")
        b2b_promo = st.file_uploader(
            "Choose Dyson Promo file",
            type=['xlsx', 'xls'],
            key='b2b_promo',
            help="Upload PromoCN Email.xlsx"
        )
        if b2b_promo:
            st.success(f"✅ {b2b_promo.name}")
    
    if st.button("🔄 Calculate B2B Support", type="primary", use_container_width=True):
        if b2b_zip and b2b_pm and b2b_promo:
            with st.spinner("Processing B2B data..."):
                result = process_b2b_data(b2b_zip, b2b_pm, b2b_promo)
                
                if result is not None:
                    st.session_state['b2b_result'] = result
                    st.markdown('<div class="success-box">✅ B2B data processed successfully!</div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please upload all three files to proceed.")
    
    # Display results
    if 'b2b_result' in st.session_state:
        result = st.session_state['b2b_result']
        
        # Key Metrics
        grand_total = result[result['Asin'] == 'Grand Total'].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Shipments", f"{int(grand_total.get('Shipment', 0)):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Net Sales", f"{int(grand_total.get('Net Sale / Actual Shipment', 0)):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Cancels", f"{int(grand_total.get('Cancel', 0)):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            support_total = grand_total.get('SUPPORT AS PER NET SALE', 0)
            st.metric("Total Support", format_currency(support_total))
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Data table
        st.markdown("### Detailed Results")
        
        # Format numeric columns for display
        display_df = result.copy()
        numeric_cols = ['SSP', 'Cons Promo', 'Support', 'SUPPORT AS PER NET SALE']
        for col in numeric_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: format_currency(x) if pd.notna(x) else '-')
        
        # Highlight Grand Total row
        def highlight_grand_total(row):
            if row['Asin'] == 'Grand Total':
                return ['background-color: #dbeafe; font-weight: bold'] * len(row)
            return [''] * len(row)
        
        st.dataframe(
            display_df.style.apply(highlight_grand_total, axis=1),
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = convert_df_to_csv(result)
        st.download_button(
            label="📥 Download B2B Results as CSV",
            data=csv,
            file_name="b2b_support_analysis.csv",
            mime="text/csv",
            use_container_width=True
        )

# B2C Tab
with tab2:
    st.markdown("### B2C Support Calculation")
    
    st.info("📁 Please upload all three required files below:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**1️⃣ B2C Report**")
        b2c_zip = st.file_uploader(
            "Choose B2C ZIP file",
            type=['zip'],
            key='b2c_zip',
            help="Upload b2cReport_October_2025.zip"
        )
        if b2c_zip:
            st.success(f"✅ {b2c_zip.name}")
    
    with col2:
        st.markdown("**2️⃣ PM File**")
        b2c_pm = st.file_uploader(
            "Choose PM Excel file",
            type=['xlsx', 'xls'],
            key='b2c_pm',
            help="Upload PM.xlsx file"
        )
        if b2c_pm:
            st.success(f"✅ {b2c_pm.name}")
    
    with col3:
        st.markdown("**3️⃣ Dyson Promo**")
        b2c_promo = st.file_uploader(
            "Choose Dyson Promo file",
            type=['xlsx', 'xls'],
            key='b2c_promo',
            help="Upload PromoCN Email.xlsx"
        )
        if b2c_promo:
            st.success(f"✅ {b2c_promo.name}")
    
    if st.button("🔄 Calculate B2C Support", type="primary", use_container_width=True):
        if b2c_zip and b2c_pm and b2c_promo:
            with st.spinner("Processing B2C data..."):
                result = process_b2c_data(b2c_zip, b2c_pm, b2c_promo)
                
                if result is not None:
                    st.session_state['b2c_result'] = result
                    st.markdown('<div class="success-box">✅ B2C data processed successfully!</div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please upload all three files to proceed.")
    
    # Display results
    if 'b2c_result' in st.session_state:
        result = st.session_state['b2c_result']
        
        # Key Metrics
        grand_total = result[result['Asin'] == 'Grand Total'].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Shipments", f"{int(grand_total.get('Shipment', 0)):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Net Sales", f"{int(grand_total.get('Net Sale / Actual Shipment', 0)):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Refunds", f"{int(grand_total.get('Refund', 0)):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            support_total = grand_total.get('SUPPORT AS PER NET SALE', 0)
            st.metric("Total Support", format_currency(support_total))
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Data table
        st.markdown("### Detailed Results")
        
        # Format numeric columns for display
        display_df = result.copy()
        numeric_cols = ['SSP', 'Cons Promo', 'Support', 'SUPPORT AS PER NET SALE']
        for col in numeric_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: format_currency(x) if pd.notna(x) else '-')
        
        # Highlight Grand Total row
        def highlight_grand_total(row):
            if row['Asin'] == 'Grand Total':
                return ['background-color: #dbeafe; font-weight: bold'] * len(row)
            return [''] * len(row)
        
        st.dataframe(
            display_df.style.apply(highlight_grand_total, axis=1),
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = convert_df_to_csv(result)
        st.download_button(
            label="📥 Download B2C Results as CSV",
            data=csv,
            file_name="b2c_support_analysis.csv",
            mime="text/csv",
            use_container_width=True
        )

# Footer with instructions
st.markdown("---")
st.markdown("### 📖 How to Use This Application")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Steps:**
    1. Select either **B2B** or **B2C** tab
    2. Upload the report ZIP file
    3. Upload the PM Excel file
    4. Upload the Dyson Promo Excel file (Dyson Promo file header should be in same format if there is error while uploading file please check header of Dyson Promo file)
    5. Click the **Calculate** button
    6. View results and download CSV
    """)

with col2:
    st.markdown("""
    **Required Files:**
    - **B2B/B2C Report**: ZIP file with CSV data
    - **PM.xlsx**: Product Master file
    - **PromoCN Email.xlsx**: Dyson Promo data
    
    **Key Calculations:**
    - **Net Sale** = Shipment - Refund
    - **Support** = (SSP - Cons Promo) × (1 - Margin%)
    - **Support Total** = Support × Net Sale
    """)

st.markdown('<div class="info-box">', unsafe_allow_html=True)
st.info("💡 **Tip:** Make sure your files are in the correct format (ZIP for reports, XLSX for promo data) before uploading.")
st.markdown('</div>', unsafe_allow_html=True)