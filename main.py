# this old code that is replace on 10/feb/2026
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


def convert_df_to_csv(df):
    """Convert dataframe to CSV for download"""
    return df.to_csv(index=False).encode('utf-8')


def format_currency(value):
    """Format number as Indian currency"""
    if pd.isna(value):
        return "-"
    return f"₹{value:,.0f}"

def get_available_months(zip_files):
    """Scan ZIP files and return unique months found in Invoice Date column"""
    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
        7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
    }
    found_months = set()
    try:
        for zip_file in zip_files:
            with zipfile.ZipFile(zip_file) as z:
                csv_files = [name for name in z.namelist() if name.endswith('.csv')]
                for csv_name in csv_files:
                    with z.open(csv_name) as f:
                        temp_df = pd.read_csv(f, usecols=["Invoice Date"])
                        dates = pd.to_datetime(temp_df["Invoice Date"], errors='coerce')
                        for m in dates.dt.month.dropna().unique():
                            found_months.add(month_names[int(m)])
            zip_file.seek(0)  # Reset file pointer so it can be read again by process_data
    except Exception:
        pass
    # Return in calendar order
    ordered = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]
    return [m for m in ordered if m in found_months]


def process_data(zip_files, pm_file, promo_file, past_months):
    """Process B2B/B2C data and calculate support"""
    try:
        # ---------- READ FILES ----------
        all_dfs = []
        for zip_file in zip_files:
            with zipfile.ZipFile(zip_file) as z:
                csv_files = [name for name in z.namelist() if name.endswith('.csv')]

                for csv_name in csv_files:
                    with z.open(csv_name) as f:
                        temp_df = pd.read_csv(f)
                        all_dfs.append(temp_df)

        df = pd.concat(all_dfs, ignore_index=True)

        # Clean and prepare data
        df["Sku"] = df["Sku"].astype(str).str.strip()
        df["Asin"] = df["Asin"].astype(str).str.strip()
        PM = pd.read_excel(pm_file)
        Promo = pd.read_excel(promo_file)
        PM["Amazon Sku Name"] = PM["Amazon Sku Name"].astype(str).str.strip()
        PM["ASIN"] = PM["ASIN"].astype(str).str.strip()
        Promo["ASIN"] = Promo["ASIN"].astype(str).str.strip()
        
        # ---------- STEP 1: BRAND MAP via SKU ----------
        brand_map_sku = PM.groupby("Amazon Sku Name", as_index=True)["Brand"].first()
        df["Brand"] = df["Sku"].map(brand_map_sku)
        
        # Move Brand column after Sku
        cols = list(df.columns)
        if "Sku" in cols and "Brand" in cols:
            sku_idx = cols.index("Sku")
            cols.remove("Brand")
            cols.insert(sku_idx + 1, "Brand")
            df = df[cols]
        
        # ---------- STEP 2: REMOVE SELECTED PAST-MONTH REFUNDS ----------
        month_map = {
            "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
            "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
        }
        
        # Parse Invoice Date
        df["Invoice Date"] = pd.to_datetime(df["Invoice Date"], errors='coerce')
        
        # Remove refund rows from user-selected past months
        if past_months:
            selected_month_nums = [month_map[m] for m in past_months]
            past_month_refund_mask = (
                (df["Invoice Date"].dt.month.isin(selected_month_nums)) &
                (df["Transaction Type"].astype(str).str.strip().str.upper() == "REFUND")
            )
            df = df[~past_month_refund_mask].copy()
        
        # ---------- DYSON ONLY ----------
        dyson_df = df[df["Brand"].notna() & (df["Brand"].astype(str).str.strip().str.upper() == "DYSON")].copy()
        
        # ---------- ORDER STATUS ----------
        cancel_orders = set(
            dyson_df[dyson_df["Transaction Type"].astype(str).str.strip().str.upper() == "CANCEL"]["Order Id"]
        )
        
        dyson_df["Order Status"] = np.where(
            dyson_df["Order Id"].isin(cancel_orders),
            "Cancel",
            dyson_df["Transaction Type"]
        )
        
        # Move Order Status after Order Id
        cols = list(dyson_df.columns)
        order_idx = cols.index("Order Id")
        cols.remove("Order Status")
        cols.insert(order_idx + 1, "Order Status")
        dyson_df = dyson_df[cols]
        
        # ---------- PROCESSED DATA (BEFORE PIVOT) ----------
        processed_df = dyson_df.copy()
        
        # ---------- PIVOT ----------
        pivot = pd.pivot_table(
            dyson_df,
            index="Asin",
            columns="Order Status",
            values="Quantity",
            aggfunc="sum",
            fill_value=0,
            margins=False
        ).reset_index()
        
        # ---------- NET SALE ----------
        pivot["Net Sale / Actual Shipment"] = (
            pivot.get("Shipment", 0) - pivot.get("Refund", 0)
        )
        
        # ---------- PROMO MAP ----------
        pivot["SKU CODE"] = pivot["Asin"].map(Promo.groupby("ASIN")["SKU Code"].first())
        pivot["SSP"] = pivot["Asin"].map(Promo.groupby("ASIN")["SSP"].first())
        pivot["Cons Promo"] = pivot["Asin"].map(Promo.groupby("ASIN")["Cons Promo"].first())
        pivot["Margin %"] = pivot["Asin"].map(Promo.groupby("ASIN")["Margin"].first()) * 100
        
        # ---------- SUPPORT ----------
        pivot["Support"] = (
            (pivot["SSP"] - pivot["Cons Promo"])
            * (1 - pivot["Margin %"] / 100)
        )
        
        pivot["SUPPORT AS PER NET SALE"] = (
            pivot["Support"].fillna(0)
            * pivot["Net Sale / Actual Shipment"].fillna(0)
        )
        
        # ---------- BASE AMOUNT (EXCLUDING 18% GST) ----------
        pivot["Base Amount"] = (pivot["SUPPORT AS PER NET SALE"] / 1.18).round(2)
        
        # ---------- CLEAN NUMERIC ----------
        pivot.replace("", np.nan, inplace=True)
        
        # Get all numeric columns (excluding Asin and SKU CODE)
        exclude_cols = ["Asin", "SKU CODE"]
        numeric_cols = [col for col in pivot.columns if col not in exclude_cols]
        
        for col in numeric_cols:
            pivot[col] = pd.to_numeric(pivot[col], errors="coerce").fillna(0)
        
        # ---------- GRAND TOTAL ----------
        grand_total = {}
        for col in pivot.columns:
            if col == "Asin":
                grand_total[col] = "Grand Total"
            elif col == "SKU CODE":
                grand_total[col] = ""
            elif col in numeric_cols:
                grand_total[col] = pivot[col].sum()
            else:
                grand_total[col] = 0
        
        pivot = pd.concat([pivot, pd.DataFrame([grand_total])], ignore_index=True)
        
        # Convert SKU CODE to string to avoid Arrow serialization issues
        pivot["SKU CODE"] = pivot["SKU CODE"].astype(str)
        
        return pivot, processed_df
        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None, None


# Main App
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 B2B Analysis", 
    "📈 B2C Analysis", 
    "🔄 Combined Analysis",
    "🧾 Invoice Qty"
])


def render_tab(tab, key):
    """Render B2B, B2C or Combined tab"""
    with tab:
        st.markdown(f"### {key} Support Calculation")
        
        st.info("📁 Please upload required files below:")
        
        if key == "Combined":
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**1️⃣ B2B Report ZIP**")
                b2b_zip = st.file_uploader(
                    "Choose B2B ZIP files",
                    type=['zip'],
                    accept_multiple_files=True,
                    key='combined_b2b_zip'
                )
                
            with col2:
                st.markdown("**2️⃣ B2C Report ZIP**")
                b2c_zip = st.file_uploader(
                    "Choose B2C ZIP files",
                    type=['zip'],
                    accept_multiple_files=True,
                    key='combined_b2c_zip'
                )
            
            col_pm, col_promo = st.columns(2)
            with col_pm:
                st.markdown("**3️⃣ PM File**")
                pm_file = st.file_uploader("Choose PM Excel file", type=['xlsx', 'xls'], key='combined_pm')
            with col_promo:
                st.markdown("**4️⃣ Dyson Promo**")
                promo_file = st.file_uploader("Choose Dyson Promo file", type=['xlsx', 'xls'], key='combined_promo')
            
            # Detect available months from uploaded ZIPs
            all_zips_for_scan = (b2b_zip if b2b_zip else []) + (b2c_zip if b2c_zip else [])
            available_months = get_available_months(all_zips_for_scan) if all_zips_for_scan else []
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**1️⃣ Report ZIP**")
                zip_files = st.file_uploader(
                    f"Choose {key} ZIP files",
                    type=['zip'],
                    accept_multiple_files=True,
                    key=f'{key}_zip'
                )
                
            with col2:
                st.markdown("**2️⃣ PM File**")
                pm_file = st.file_uploader("Choose PM Excel file", type=['xlsx', 'xls'], key=f'{key}_pm')
                
            with col3:
                st.markdown("**3️⃣ Dyson Promo**")
                promo_file = st.file_uploader("Choose Dyson Promo file", type=['xlsx', 'xls'], key=f'{key}_promo')
            
            # Detect available months from uploaded ZIPs
            available_months = get_available_months(zip_files) if zip_files else []
        
        # Show past months multiselect only if ZIP files are uploaded
        if available_months:
            past_months = st.multiselect(
                f"Select past months to remove Refund data ({key})",
                options=available_months,
                default=[],
                key=f'past_months_{key}',
                help="These months were found in the Invoice Date column of your uploaded files. Select which ones to remove Refund data from."
            )
        else:
            past_months = []
        
        # Calculate button
        if key == "Combined":
            if st.button("🔄 Calculate Combined Support", type="primary", use_container_width=True):
                if (b2b_zip or b2c_zip) and pm_file and promo_file:
                    all_zips = (b2b_zip if b2b_zip else []) + (b2c_zip if b2c_zip else [])
                    with st.spinner("Processing combined data..."):
                        pivot, processed = process_data(all_zips, pm_file, promo_file, past_months)
                        if pivot is not None:
                            st.session_state[f'{key}_pivot'] = pivot
                            st.session_state[f'{key}_processed'] = processed
                            st.success("✅ Combined data processed successfully!")
                else:
                    st.warning("⚠️ Please upload at least one report ZIP and both PM/Promo files.")
        else:
            if st.button(f"🔄 Calculate {key} Support", type="primary", use_container_width=True):
                if zip_files and pm_file and promo_file:
                    with st.spinner(f"Processing {key} data..."):
                        pivot, processed = process_data(zip_files, pm_file, promo_file, past_months)
                        if pivot is not None:
                            st.session_state[f'{key}_pivot'] = pivot
                            st.session_state[f'{key}_processed'] = processed
                            st.success(f"✅ {key} data processed successfully!")
                else:
                    st.warning("⚠️ Please upload all three files to proceed.")
        
        # -------- PROCESSED DATA --------
        if f'{key}_processed' in st.session_state:
            st.markdown("---")
            st.markdown("### 🧾 Processed Dyson Data (Before Pivot)")
            st.dataframe(
                st.session_state[f'{key}_processed'],
                height=350,
                use_container_width=True
            )
            
            csv = convert_df_to_csv(st.session_state[f'{key}_processed'])
            st.download_button(
                label="📥 Download Processed Data (Before Pivot)",
                data=csv,
                file_name=f"{key.lower()}_processed_before_pivot.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.markdown("---")
        
        # -------- FINAL RESULT --------
        if f'{key}_pivot' in st.session_state:
            result = st.session_state[f'{key}_pivot']
            
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
                metric_label = "Total Cancels" if key == "B2B" else "Total Refunds"
                metric_value = grand_total.get('Cancel', 0) if key == "B2B" else grand_total.get('Refund', 0)
                st.metric(metric_label, f"{int(metric_value):,}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                support_total = grand_total.get('SUPPORT AS PER NET SALE', 0)
                st.metric("Total Support", format_currency(support_total))
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Data table
            st.markdown("### 📊 Final Support Calculation")
            
            # Format numeric columns for display
            display_df = result.copy()
            numeric_cols = ['SSP', 'Cons Promo', 'Support', 'SUPPORT AS PER NET SALE','Base Amount']
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
                label=f"📥 Download {key} Final Results as CSV",
                data=csv,
                file_name=f"{key.lower()}_final_support_analysis.csv",
                mime="text/csv",
                use_container_width=True
            )

# Render tabs
render_tab(tab1, "B2B")
render_tab(tab2, "B2C")
render_tab(tab3, "Combined")

# ---------------- INVOICE QTY REPORT ----------------

with tab4:
    st.markdown("### 🧾 Invoice Qty Report")
    st.info("Upload Invoice file and Promo CN file to generate report")

    col1, col2 = st.columns(2)

    with col1:
        invoice_file = st.file_uploader(
            "Upload Invoice Excel File",
            type=['xlsx', 'xls'],
            key="invoice_qty_file"
        )

    with col2:
        promo_file_invoice = st.file_uploader(
            "Upload Promo CN File",
            type=['xlsx', 'xls'],
            key="invoice_promo_file"
        )
    
    handling_rate = st.number_input(
        "Enter Handling Charges (₹ per Qty)",
        min_value=0.0,
        value=270.0,
        step=10.0
    )

    if st.button("🔄 Generate Invoice Qty Report", type="primary", use_container_width=True):
        if invoice_file is not None and promo_file_invoice is not None:
            try:
                df_invoice = pd.read_excel(invoice_file)
                df_invoice.columns = df_invoice.columns.str.strip()

                promo_df = pd.read_excel(promo_file_invoice)
                promo_df.columns = promo_df.columns.str.strip()
                
                # ---- STEP 1: BASIC PIVOT ----
                pivot_invoice = pd.pivot_table(
                    df_invoice,
                    index="Material_Cd",
                    values=["Qty", "Total_Val"],
                    aggfunc="sum",
                    fill_value=0,
                    margins=True,
                    margins_name="Grand Total"
                ).reset_index()
                
                df_invoice=pivot_invoice.copy()

                # ---- CREATE CONSUMER PROMO (VLOOKUP Equivalent) ----
                # Excel: VLOOKUP(Material_Code, D:L, 9, 0)
                # Means:
                # Lookup value = Material_Code
                # Lookup column = Column D in promo file
                # Return column = 9th column from D (D=1 → L=9)

                lookup_column = promo_df.columns[3]   # Column D
                return_column = promo_df.columns[11]  # Column L (D to L = 9th column)

                promo_map = promo_df.set_index(lookup_column)[return_column]

                df_invoice["Consumer Promo"] = df_invoice["Material_Cd"].map(promo_map)
                
                # ---------- CALCULATIONS ----------

                # Total Amount
                df_invoice["Total Amount"] = df_invoice["Consumer Promo"].fillna(0) * df_invoice["Qty"]

                # 1% CN
                df_invoice["1% CN"] = df_invoice["Total Amount"] * 0.01

                # Without GST (CN Base)
                df_invoice["Without GST (CN Base)"] = df_invoice["1% CN"] / 1.18

                # Wt Handling
                df_invoice["Wt Handling"] = handling_rate * df_invoice["Qty"]

                # Without GST per Handling
                df_invoice["Without GST per Handling"] = df_invoice["Wt Handling"] / 1.18

                # Total (1% CN + Wt Handling)
                df_invoice["Total"] = df_invoice["1% CN"] + df_invoice["Wt Handling"]

                # Total Base
                df_invoice["Total Base"] = df_invoice["Total"] / 1.18

                
                desired_order = ["Material_Cd", "Qty", "Total_Val", "Consumer Promo", "Total Amount", "1% CN", "Without GST (CN Base)", "Wt Handling", "Without GST per Handling", "Total", "Total Base"]
                df_invoice = df_invoice[desired_order]

                st.success("✅ Invoice Qty Report Generated Successfully!")

                st.markdown("### 📊 Pivot Table")
                st.dataframe(df_invoice, use_container_width=True, height=400)

                csv = convert_df_to_csv(df_invoice)
                st.download_button(
                    label="📥 Download Invoice Qty Report",
                    data=csv,
                    file_name="invoice_qty_report.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        else:
            st.warning("⚠️ Please upload both Invoice file and Promo CN file.")


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
    6. View processed data (before pivot) and final results
    7. Download CSV files as needed
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
