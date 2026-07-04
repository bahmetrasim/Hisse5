import streamlit as st
import yfinance as yf
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

# Z-Skorunun 3.0'ın üzerinde olması şirketin "Güvenli Bölge"de (Safe Zone) olduğunu,
# 1.8 ile 3.0 arasının riskin başladığı "Gri Bölge"yi (Grey Zone),
# 1.8'in altının ise yüksek iflas riskini (Distress) temsil ettiğini göstermektedi

# F-Skoru 7, 8 veya 9 olan 50 USD altındaki hisseler, piyasanın korkusuna rağmen olağanüstü defansif yapılara ve iyileşen bir iş modeline sahiptir.
# P/B ortalama S&P500 => 5 Midcap 400 => 2.5

# --- İNDİKATÖR & FİNANSAL FONKSİYONLAR (SAF PANDAS İLE) ---
def hesapla_rsi(veri, periyot=14):
    fark = veri.diff()
    yukari = fark.clip(lower=0)
    asagi = -1 * fark.clip(upper=0)
    ema_yukari = yukari.ewm(com=periyot-1, adjust=False).mean()
    ema_asagi = asagi.ewm(com=periyot-1, adjust=False).mean()
    rs = ema_yukari / ema_asagi
    return 100 - (100 / (1 + rs))

def hesapla_stokastik(yuksek, dusuk, kapanis, periyot=14):
    en_dusuk = dusuk.rolling(window=periyot).min()
    en_yuksek = yuksek.rolling(window=periyot).max()
    return 100 * ((kapanis - en_dusuk) / (en_yuksek - en_dusuk))

def hesapla_mfi(df, periyot=14):
    tipik_fiyat = (df['High'] + df['Low'] + df['Close']) / 3
    para_akisi = tipik_fiyat * df['Volume']
    
    fark = tipik_fiyat.diff()
    pozitif_akis = para_akisi.where(fark > 0, 0)
    negatif_akis = para_akisi.where(fark < 0, 0)
    
    poz_toplam = pozitif_akis.rolling(window=periyot).sum()
    neg_toplam = negatif_akis.rolling(window=periyot).sum()
    
    mfi_oran = poz_toplam / neg_toplam
    mfi = 100 - (100 / (1 + mfi_oran))
    return mfi

def hesapla_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd_hist

def hesapla_bbands(close, length=20, std=2.0):
    sma = close.rolling(window=length).mean()
    std_dev = close.rolling(window=length).std()
    bbu = sma + (std_dev * std)
    bbl = sma - (std_dev * std)
    return bbu, bbl

def hesapla_keltner(df, length=20, scalar=1.5):
    ema = df['Close'].ewm(span=length, adjust=False).mean()
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift(1)).abs()
    tr3 = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=length).mean()
    kcu = ema + (scalar * atr)
    kcl = ema - (scalar * atr)
    return kcu, kcl

def hesapla_piotroski(ticker_obj):
    try:
        bs = ticker_obj.balance_sheet
        cf = ticker_obj.cashflow
        ic = ticker_obj.income_stmt
        if bs.empty or cf.empty or ic.empty: return 0
        
        score = 0
        def get_val(df, keys, col_idx):
            for k in keys:
                if k in df.index:
                    try: return float(df.loc[k].iloc[col_idx])
                    except: pass
            return 0
            
        ni = get_val(ic, ['Net Income', 'Net Income Continuous Operations'], 0)
        ni_prev = get_val(ic, ['Net Income', 'Net Income Continuous Operations'], 1)
        ta_val = get_val(bs, ['Total Assets'], 0) or 1
        ta_prev = get_val(bs, ['Total Assets'], 1) or 1
        cfo = get_val(cf, ['Operating Cash Flow', 'Total Cash From Operating Activities'], 0)
        
        roa = ni / ta_val
        roa_prev = ni_prev / ta_prev
        if roa > 0: score += 1
        if cfo > 0: score += 1
        if roa > roa_prev: score += 1
        if cfo > ni: score += 1
        
        ltd = get_val(bs, ['Long Term Debt'], 0)
        ltd_prev = get_val(bs, ['Long Term Debt'], 1)
        if (ltd/ta_val) < (ltd_prev/ta_prev): score += 1
        
        ca = get_val(bs, ['Current Assets'], 0)
        ca_prev = get_val(bs, ['Current Assets'], 1)
        cl = get_val(bs, ['Current Liabilities'], 0) or 1
        cl_prev = get_val(bs, ['Current Liabilities'], 1) or 1
        if (ca/cl) > (ca_prev/cl_prev): score += 1
        
        shares = get_val(bs, ['Ordinary Shares Number', 'Share Issued'], 0)
        shares_prev = get_val(bs, ['Ordinary Shares Number', 'Share Issued'], 1)
        if shares <= shares_prev: score += 1
        
        gp = get_val(ic, ['Gross Profit'], 0)
        gp_prev = get_val(ic, ['Gross Profit'], 1)
        tr = get_val(ic, ['Total Revenue'], 0) or 1
        tr_prev = get_val(ic, ['Total Revenue'], 1) or 1
        if (gp/tr) > (gp_prev/tr_prev): score += 1
        if (tr/ta_val) > (tr_prev/ta_prev): score += 1
        
        return score
    except:
        return 0

def hesapla_altman_z(ticker_obj, market_cap):
    try:
        bs = ticker_obj.balance_sheet
        ic = ticker_obj.income_stmt
        if bs.empty or ic.empty: return 0
        
        def get_val(df, keys):
            for k in keys:
                if k in df.index:
                    try: return float(df.loc[k].iloc[0])
                    except: pass
            return 0
            
        ta_val = get_val(bs, ['Total Assets']) or 1
        ca_val = get_val(bs, ['Current Assets'])
        cl_val = get_val(bs, ['Current Liabilities'])
        re_val = get_val(bs, ['Retained Earnings'])
        ebit_val = get_val(ic, ['EBIT'])
        tl_val = get_val(bs, ['Total Liabilities Net Minority Interest', 'Total Liabilities']) or 1
        tr_val = get_val(ic, ['Total Revenue'])
        
        wc = ca_val - cl_val
        x1 = wc / ta_val
        x2 = re_val / ta_val
        x3 = ebit_val / ta_val
        x4 = market_cap / tl_val
        x5 = tr_val / ta_val
        
        z = (1.2 * x1) + (1.4 * x2) + (3.3 * x3) + (0.6 * x4) + (1.0 * x5)
        return z
    except:
        return 0

# --- SABİT LİSTE ---
FULL_LIST = [
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "AVGO", "TSLA", "META", "MU", "LLY", "BRK.B", "AMD", "WMT", "JPM", "INTC", "V", "JNJ", "AMAT", "XOM", "LRCX", "CAT", "CSCO", "MA", "ABBV", "ORCL", "COST", "BAC", "KLAC", "GE", "UNH", "KO", "HD", "PG", "CVX", "SNDK", "MS", "MRK", "GEV", "NFLX", "GS", "PM", "PLTR", "PANW", "DELL", "TXN", "IBM", "MRVL", "WFC", "RTX", "LIN", "C", "AXP", "WDC", "APH", "GLW", "ANET", "STX", "QCOM", "AMGN", "ADI", "CRWD", "MCD", "PEP", "TMO", "NEE", "TMUS", "VZ", "APP", "DE", "BA", "DIS", "TJX", "ETN", "UNP", "WELL", "SCHW", "ABT", "GILD", "BLK", "UBER", "T", "ISRG", "BX", "HON", "BKNG", "PFE", "DHR", "CB", "CVS", "PGR", "CRM", "PLD", "VRT", "COP", "VRTX", "COF", "LOW", "PH", "MO", "SYK", "SPGI", "BMY", "SBUX", "LMT", "FTNT", "SO", "TT", "PWR", "HWM", "EQIX", "CDNS", "MDT", "NOW", "NEM", "BNY", "DUK", "PNC", "CMI", "MAR", "GD", "USB", "MNST", "DDOG", "WMB", "UPS", "FCX", "HOOD", "JCI", "WM", "ADP", "CEG", "CSX", "MCK", "CMCSA", "HCA", "ABNB", "RCL", "SNPS", "ELV", "SHW", "MMM", "KKR", "DASH", "ADBE", "EMR", "MRSH", "CME", "MCO", "ECL", "VLO", "ITW", "AMT", "ORLY", "COHR", "ACN", "HLT", "MDLZ", "MPC", "AEP", "TER", "FDX", "TDG", "CI", "CL", "SPG", "KMI", "NOC", "CRH", "INTU", "NSC", "AON", "NXPI", "URI", "TRV", "ICE", "EOG", "SLB", "GM", "FIX", "MSI", "PSX", "CIEN", "ROST", "CTAS", "LITE", "MPWR", "WBD", "APO", "RSG", "APD", "REGN", "GWW", "PCAR", "DLR", "BSX", "TFC", "ALL", "NKE", "DAL", "CARR", "SRE", "D", "KEYS", "AFL", "FLEX", "TGT", "TEL", "HPE", "TRGP", "AJG", "O", "CTVA", "PSA", "CAH", "OKE", "BKR", "F", "AME", "FAST", "COR", "ROK", "MET", "LHX", "ETR", "VST", "EW", "AZO", "EA", "FITB", "NUE", "XEL", "FANG", "MCHP", "EBAY", "OXY", "EXC", "DVN", "HUM", "STT", "TTWO", "CVNA", "DHI", "WAB", "GRMN", "XYZ", "KDP", "ODFL", "NDAQ", "AXON", "UAL", "YUM", "VTR", "CCL", "CMG", "LYV", "BDX", "IDXX", "ED", "AMP", "PEG", "ADSK", "MSCI", "JBL", "AIG", "SYY", "IBKR", "CBRE", "WEC", "COIN", "VMC", "PYPL", "IRM", "PRU", "PCG", "A", "ADM", "EME", "KVUE", "ON", "WAT", "KMB", "HIG", "HBAN", "HSY", "PAYX", "MTB", "ACGL", "MLM", "ROP", "Q", "KR", "CCI", "EQT", "STLD", "NTRS", "IR", "BIIB", "IQV", "DTE", "CNC", "AEE", "EXPE", "NRG", "EXR", "TDY", "LVS", "DOV", "NTAP", "ZTS", "WDAY", "SATS", "TPL", "TPR", "CFG", "RJF", "CASY", "CNP", "GEHC", "EIX", "ATO", "CINF", "VICI", "VEEV", "HAL", "EL", "KHC", "RMD", "MRNA", "XYL", "WSM", "FE", "PPL", "FICO", "HUBB", "ES", "OTIS", "JBHT", "PPG", "AVB", "WRB", "DXCM", "PHM", "AWK", "RF", "FISV", "CPRT", "MTD", "SYF", "EQR", "DG", "FSLR", "CBOE", "WST", "LUV", "KEY", "ARES", "WTW", "TROW", "SW", "RL", "CMS", "FFIV", "DGX", "VRSK", "DRI", "L", "PFG", "DLTR", "STZ", "LH", "CHD", "NI", "VRSN", "FDXF", "INCY", "LEN", "CHRW", "EXE", "VLTO", "BRO", "CPAY", "EXPD", "PKG", "BG", "SNA", "OMC", "STE", "HPQ", "AMCR", "TSN", "IP", "EVRG", "ROL", "LII", "FIS", "LNT", "DOW", "IFF", "GPN", "ULTA", "SMCI", "SBAC", "ESS", "VTRS", "EFX", "GIS", "FTV", "DD", "NVR", "CTSH", "INVH", "CDW", "BEN", "KIM", "GNRC", "WY", "AKAM", "CHTR", "LYB", "NDSN", "CF", "IEX", "BALL", "TSCO", "MAS", "HST", "MAA", "ZBH", "GPC", "ALB", "TXT", "BBY", "BR", "TKO", "GEN", "DOC", "J", "REG", "SWK", "DVA", "EG", "COO", "GL", "DECK", "HRL", "MKC", "AIZ", "SOLV", "PNW", "PTC", "UDR", "APTV", "LULU", "LDOS", "AVY", "ERIE", "BF.B", "PNR", "ZBRA", "RVTY", "MGM", "SJM", "ALLE", "TYL", "ALGN", "TRMB", "IVZ", "HAS", "CSGP", "APA", "CPT", "CLX", "GDDY", "PSKY", "HII", "BAX", "TECH", "CRL", "FRT", "BXP", "PODD", "AES", "SWKS", "FOXA", "FOX", "WYNN", "DPZ", "JKHY", "NCLH", "BLDR", "HSIC", "ARE", "NWSA", "UHS", "IT", "AOS", "TTD", "FDS", "TAP", "MOS", "CAG", 
    "NWS", "MTZ", "TWLO", "CRS", "MTSI", "MKSI", "ILMN", "CW", "ATI", "NVT", "ENTG", "FTI", "WWD", "ROIV", "STRL", "XPO", "P", "UTHR", "OKTA", "USFD", "SNX", "DKS", "SN", "AMKR", "ROKU", "ULS", "LSCC", "RBC", "BURL", "TTMI", "RS", "FN", "SITM", "H", "TLN", "APG", "PFGC", "EWBC", "ONTO", "BWXT", "RGLD", "NBIX", "ITT", "NLY", "CDE", "WCC", "VICR", "WSO", "NXT", "SGI", "WPC", "THC", "LAMR", "CLH", "DOCN", "PR", "TOL", "CSL", "MEDP", "VNOM", "PNFP", "DY", "DTM", "OVV", "JAZZ", "ARMK", "RRX", "CG", "SMTC", "JLL", "ALLY", "LECO", "UNM", "OHI", "CNH", "RPM", "AA", "RGA", "WMG", "TRU", "EVR", "NTNX", "RNR", "EXEL", "MLI", "BWA", "MOG.A", "RMBS", "AEIS", "CRBG", "SOLS", "GLPI", "DT", "SANM", "FNF", "COKE", "DINO", "TXRH", "CR", "KNX", "GGG", "OC", "ELS", "BROS", "EQH", "CCK", "PEN", "ALGM", "ELAN", "AIT", "WBS", "PINS", "AMH", "FHN", "WTS", "SPXC", "AAL", "WMS", "LAD", "CYTK", "VIAV", "AFG", "NYT", "BJ", "GMED", "CGNX", "LFUS", "SAIA", "ARWR", "BMRN", "AYI", "VMI", "EGP", "CART", "ARW", "WTRG", "AM", "WTFC", "UMBF", "AR", "SCI", "SF", "SEIC", "HL", "RYAN", "AAON", "DCI", "R", "ZION", "OGE", "ORI", "CACI", "BLD", "GWRE", "EHC", "GTLS", "JEF", "TKR", "ONB", "GME", "MUSA", "BRX", "FIVE", "SIRI", "SSB", "MP", "CFR", "OSK", "WLK", "CTRE", "CAVA", "SARO", "FCFS", "FLS", "TTC", "CNM", "COLB", "ENSG", "ADC", "CUBE", "HLI", "HALO", "BRKR", "WAL", "AMG", "BSY", "NNN", "ACM", "PRI", "KTOS", "AGCO", "SSD", "DOCU", "ALV", "RRC", "CBSH", "DAR", "IDA", "TEX", "FR", "VOYA", "ENS", "MANH", "VLY", "JHG", "ATR", "CHWY", "BIO", "MIDD", "HIMS", "RGEN", "REXR", "SFM", "KNSL", "CELH", "THG", "FLR", "TTEK", "STAG", "UGI", "NFG", "HXL", "BAH", "VNO", "PB", "CRUS", "KEX", "HQY", "AXTA", "NEU", "SLAB", "AVT", "CMC", "IDCC", "HR", "LNTH", "LSTR", "AVAV", "EXP", "FAF", "PPC", "AVTR", "FNB", "ST", "ORA", "GBCI", "GAP", "LEA", "ACI", "BYD", "LAD", "MSA", "NOV", "TMHC", "VFC", "CHRD", "RYN", "MSM", "WH", "SWX", "M", "CHDN", "GATX", "FND", "AN", "TXNM", "UBSI", "STWD", "FLG", "CROX", "MTDR", "DBX", "CHE", "FBIN", "HWC", "POR", "INGR", "ESAB", "ESNT", "MTG", "CVLT", "MORN", "KRG", "SIGI", "WFRD", "GXO", "HOMB", "ALK", "NJR", "BKH", "PCTY", "OZK", "LPX", "NOVT", "BC", "APPF", "PATH", "SON", "PBF", "PSN", "CAR", "DUOL", "RLI", "GNTX", "CLF", "AHR", "NXST", "UFPI", "TREX", "VAL", "ASB", "CHH", "PEGA", "VVV", "SHC", "GHC", "CUZ", "FFIN", "OGS", "DLB", "SLM", "SBRA", "SLGN", "KNF", "CNO", "MUR", "TNL", "HRB", "MTN", "SAIC", "WEX", "BBWI", "IBOC", "G", "CNX", "IPGP", "SR", "CBT", "BDC", "LIVN", "SYNA", "QLYS", "EPR", "WING", "FCN", "TCBI", "KRC", "NWE", "NVST", "OLLI", "FHI", "KBR", "HGV", "HLNE", "ELF", "CDP", "THO", "PLNT", "POST", "PII", "VNT", "OLED", "MAT", "ANF", "IRT", "SMG", "KBH", "EXLS", "FOUR", "YETI", "BCO", "DOCS", "LOPE", "BHF", "BILL", "NSA", "GEF", "HAE", "PVH", "AVNT", "OPCH", "MZTI", "COLM", "GPK", "RH", "ASH", "PK", "EXPO", "MMS", "CXT", "EEFT", "HOG", "VC", "KD", "WHR", "OLN", "CPRI", "XRAY", "GT", "SAM"
]

# --- TARAMA MOTORU ---
st.set_page_config(page_title="Anayasa v17", layout="centered")
st.title("Yatırım Anayasası v17 (Özel Sembol Destekli)")

# KULLANICI GİRDİSİ 
ek_semboller_metni = st.text_input(
    "Listeye Eklemek İstediğiniz Semboller (Virgülle ayırın):", 
    placeholder="Örn: WDOFF, PLTR, BIST:THYAO"
)

# YAN YANA İKİ BUTON TASARIMI
col1, col2 = st.columns(2)
with col1:
    btn_sadece_kutu = st.button("Sadece Kutudakileri Tara", use_container_width=True)
with col2:
    btn_tum_liste = st.button("Kutu + Full Listeyi Tara", use_container_width=True)

# İki butondan herhangi birine basılırsa tarama tetiklenir
if btn_sadece_kutu or btn_tum_liste:
    
    # Metin kutusundaki veriyi temizleyip listeye çevir
    girilen_semboller = [s.strip().upper() for s in ek_semboller_metni.split(',') if s.strip()]
    
    # Hangi butonun basıldığına göre nihai listeyi belirle
    if btn_sadece_kutu:
        nihai_liste = list(dict.fromkeys(girilen_semboller))
        if len(nihai_liste) == 0:
            st.warning("⚠️ Lütfen taramak için kutuya en az bir sembol girin!")
            st.stop() # Hata vermemek için kodun aşağıya inmesini engeller
    else:
        # Full liste butonu basıldıysa ikisini birleştirip kopyaları sil
        nihai_liste = list(dict.fromkeys(girilen_semboller + FULL_LIST))
    
    st.info(f"📊 Toplam Taranacak Hisse Sayısı: {len(nihai_liste)}")

    uygunlar1, uygunlar2, uygunlar3, uygunlar4, uygunlar5, uygunlar6 = [], [], [], [], [], []
    paket_sayisi = len(nihai_liste) // 50 + (1 if len(nihai_liste) % 50 != 0 else 0)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, len(nihai_liste), 50):
        paket = nihai_liste[i:i+50]
        mevcut_paket_no = i//50 + 1
        status_text.write(f"⏳ İşleniyor: Paket {mevcut_paket_no} / {paket_sayisi} taranıyor...")
        
        for ticker in paket:
            try:
                time.sleep(0.5) 
                
                # Yahoo Finance formatı düzeltmesi: Noktaları tireye çevir 
                yf_ticker = ticker.replace('.', '-')
                
                ticker_obj = yf.Ticker(yf_ticker)
                
                df = ticker_obj.history(period="6mo")
                info = ticker_obj.info
                
                if len(df) < 50: continue
                
                close = df['Close']
                son_kapanis = float(close.iloc[-1])
                
                # Fiyat filtresi
                if son_kapanis > 50: continue
                
                # --- TEMEL VERİLER ---
                pb = info.get('priceToBook', 999)
                peg_raw = info.get('pegRatio')
                peg = peg_raw if peg_raw is not None else 0.5
                ev_ebitda = info.get('enterpriseToEbitda', 999)
                ev_ebitda = ev_ebitda if ev_ebitda is not None else 999
                market_cap = info.get('marketCap', 1)
                fcf = info.get('freeCashflow', 0)
                fcf_yield = fcf / market_cap if market_cap else 0
                short_float = info.get('shortPercentOfFloat', 0)
                short_float = short_float if short_float is not None else 0
                insiders = info.get('heldPercentInsiders', 0)
                insiders = insiders if insiders is not None else 0
                target_price = info.get('targetMeanPrice', 0)
                target_price = target_price if target_price is not None else 0

                # --- ORTAK TEKNİK VERİLER (TABLOLAR İÇİN DE KULLANILACAK) ---
                high = df['High']
                low = df['Low']
                rsi_serisi = hesapla_rsi(close)
                stoch_serisi = hesapla_stokastik(high, low, close)
                ema9_serisi = close.ewm(span=9, adjust=False).mean()
                ema21_serisi = close.ewm(span=21, adjust=False).mean()
                
                # Son gün verileri
                son_rsi = round(float(rsi_serisi.iloc[-1]), 2)
                son_stoch = round(float(stoch_serisi.iloc[-1]), 2)
                son_ema9 = round(float(ema9_serisi.iloc[-1]), 2)
                
                # --- UYGUNLAR 1 & 2 ---
                if (0 < pb <= 1.5) and (0 < peg <= 1) and (son_rsi <= 40) and (son_stoch <= 20):
                    uygunlar1.append({
                        "Sembol": ticker, "Fiyat": son_kapanis, "P/B": pb, "PEG": peg,
                        "RSI": son_rsi, "Stoch": son_stoch, "EMA 9": son_ema9
                    })
                
                if (0 < pb <= 1.5) and (0 < peg <= 1) and \
                   (float(ema9_serisi.iloc[-2]) <= float(ema21_serisi.iloc[-2])) and (float(ema9_serisi.iloc[-1]) > float(ema21_serisi.iloc[-1])) and \
                   (rsi_serisi.iloc[-7:].min() <= 40) and (stoch_serisi.iloc[-7:].min() <= 20):
                    uygunlar2.append({
                        "Sembol": ticker, "Fiyat": son_kapanis, "P/B": pb, "PEG": peg,
                        "RSI": son_rsi, "Stoch": son_stoch, "EMA 9": son_ema9
                    })

                # --- UYGUNLAR 3 (Liste 1) ---
                if (ev_ebitda <= 10) and (fcf_yield >= 0.08):
                    mfi_s = hesapla_mfi(df)
                    if not mfi_s.empty and mfi_s.iloc[-1] <= 20:
                        piotroski_score = hesapla_piotroski(ticker_obj)
                        if piotroski_score >= 7:
                            uygunlar3.append({
                                "Sembol": ticker, "Fiyat": son_kapanis, "EV/EBITDA": round(ev_ebitda, 2), 
                                "FCF Y.": round(fcf_yield, 3), "F-Skor": piotroski_score,
                                "RSI": son_rsi, "Stoch": son_stoch, "EMA 9": son_ema9
                            })

                # --- UYGUNLAR 4 (Liste 2) ---
                if short_float >= 0.15 and (insiders >= 0.05 or target_price >= son_kapanis * 1.30):
                    bbu, bbl = hesapla_bbands(close)
                    kcu, kcl = hesapla_keltner(df)
                    
                    if not bbu.empty and not kcu.empty:
                        squeeze_on = (bbu < kcu) & (bbl > kcl)
                        squeeze_fired = (~squeeze_on.iloc[-1]) and (squeeze_on.iloc[-2] or squeeze_on.iloc[-3])
                        momentum_pozitif = son_kapanis > close.rolling(20).mean().iloc[-1]
                        
                        if squeeze_fired and momentum_pozitif:
                            uygunlar4.append({
                                "Sembol": ticker, "Fiyat": son_kapanis, "Short Float": round(short_float, 2), 
                                "Insiders": round(insiders, 3),
                                "RSI": son_rsi, "Stoch": son_stoch, "EMA 9": son_ema9
                            })
                
                # --- UYGUNLAR 5 (Liste 3) ---
                if (float(ema9_serisi.iloc[-2]) <= float(ema21_serisi.iloc[-2])) and (float(ema9_serisi.iloc[-1]) > float(ema21_serisi.iloc[-1])):
                    macdh = hesapla_macd(close)
                    if not macdh.empty:
                        son_20 = df[-20:]
                        onceki_20 = df[-40:-20]
                        macdh_son20 = macdh[-20:]
                        macdh_onceki20 = macdh[-40:-20]
                        
                        if (son_20['Close'].min() < onceki_20['Close'].min()) and (macdh_son20.min() > macdh_onceki20.min()):
                            z_score = hesapla_altman_z(ticker_obj, market_cap)
                            if z_score >= 3.0:
                                uygunlar5.append({
                                    "Sembol": ticker, "Fiyat": son_kapanis, "Z-Skor": round(z_score, 2),
                                    "RSI": son_rsi, "Stoch": son_stoch, "EMA 9": son_ema9
                                })
                                
                # --- UYGUNLAR 6: Esnek Fırsat Avcısı ---
                if (0 < pb <= 2.5) and (0 < peg <= 1.5) and (ev_ebitda <= 15) and (son_rsi <= 50):
                    p_skor = hesapla_piotroski(ticker_obj)
                    z_skor = hesapla_altman_z(ticker_obj, market_cap)
                    
                    if p_skor >= 5 and z_skor >= 1.8:
                        uygunlar6.append({
                            "Sembol": ticker, "Fiyat": son_kapanis, "P/B": round(pb, 2), 
                            "PEG": round(peg, 2), "EV/EB": round(ev_ebitda, 1),
                            "F-Skor": p_skor, "Z-Skor": round(z_skor, 2),
                            "RSI": son_rsi, "Stoch": son_stoch, "EMA 9": son_ema9
                        })
                        
            except: continue
        time.sleep(5)
        progress_bar.progress(mevcut_paket_no / paket_sayisi)
    
    status_text.empty()
    progress_bar.empty()
    st.success("✅ Tarama Tamamlandı!")
    
    st.subheader(f"1. Temel Değer Odaklı ({len(uygunlar1)} Hisse)")
    st.dataframe(pd.DataFrame(uygunlar1), width='stretch')
    
    st.subheader(f"2. Temel + Teknik Trend Dönüşü ({len(uygunlar2)} Hisse)")
    st.dataframe(pd.DataFrame(uygunlar2), width='stretch')
    
    st.subheader(f"3. Gizli Nakit Devi & Değer Avcısı - Liste 1 ({len(uygunlar3)} Hisse)")
    st.dataframe(pd.DataFrame(uygunlar3), width='stretch')
    
    st.subheader(f"4. Kısa Sıkıştırması ve Volatilite - Liste 2 ({len(uygunlar4)} Hisse)")
    st.dataframe(pd.DataFrame(uygunlar4), width='stretch')
    
    st.subheader(f"5. Zırhlı Trend Dönüşü - Liste 3 ({len(uygunlar5)} Hisse)")
    st.dataframe(pd.DataFrame(uygunlar5), width='stretch')
    
    st.subheader(f"6. Esnek Fırsat Avcısı (Yumuşatılmış Kombinasyon) ({len(uygunlar6)} Hisse)")
    st.dataframe(pd.DataFrame(uygunlar6), width='stretch')
