import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np
import os
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
def build_lstm_model(input_shape, lstm_units, dense_units, learning_rate):
    model = Sequential()
    for i, units in enumerate(lstm_units):
        return_sequences = i < len(lstm_units) - 1  # False for last LSTM layer
        if(i == 0):
            model.add(LSTM(units, return_sequences=return_sequences, input_shape=input_shape))
        else:
            model.add(LSTM(units, return_sequences=return_sequences))
    for units in dense_units:
        model.add(Dense(units))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model
domain_of_activity = {
    "Technology": 1,
    "Healthcare": 2,
    "Finance": 3,
    "Retail": 4,
    "Manufacturing": 5,
    "Energy": 6,
    "Telecommunications": 7,
    "Automotive": 8,
    "Consumer Goods": 9,
    "Real Estate": 10,
    "Transportation": 11,
    "Agriculture": 12,
    "Education": 13,
    "Media": 14,
    "Hospitality": 15
}
companies_domains = {
    "TTE": "Energy",            # TotalEnergies SE
    "AIR.PA": "Manufacturing",  # Airbus SE
    "SNY": "Healthcare",        # Sanofi
    "SU.PA": "Technology",      # Schneider Electric
    "AI.PA": "Manufacturing",   # Air Liquide
    "EL.PA": "Healthcare",      # EssilorLuxottica
    "BNP.PA": "Finance",        # BNP Paribas
    "SAF.PA": "Manufacturing",  # Safran
    "CS.PA": "Finance",         # AXA
    "DG.PA": "Construction",    # Vinci
    "STLA": "Automotive",       # Stellantis
    "DSY.PA": "Technology",     # Dassault Systèmes
    "KER.PA": "Consumer Goods", # Kering
    "RI.PA": "Consumer Goods",  # Pernod Ricard
    "STM": "Technology",        # STMicroelectronics
    "ENGI.PA": "Energy",        # ENGIE
    "ACA.PA": "Finance",        # Crédit Agricole
    "BN.PA": "Consumer Goods",  # Danone
    "CAP.PA": "Technology",     # Capgemini
    "SGO.PA": "Manufacturing",  # Compagnie de Saint-Gobain
    "ORAN": "Telecommunications",   # Orange SA
    "HO.PA": "Technology",      # Thales SA
    "LR.PA": "Manufacturing",   # Legrand SA
    "ML.PA": "Automotive",      # Michelin
    "PUB.PA": "Media",          # Publicis Groupe SA
    "VIE.PA": "Utilities",      # Veolia Environnement SA
    "MT": "Manufacturing",      # ArcelorMittal
    "CA.PA": "Retail",          # Carrefour SA
    "GLE.PA": "Finance",        # Société Générale SA
    "OR.PA": "Consumer Goods",  # L'Oréal
    "RMS.PA": "Consumer Goods", # Hermès International
    "MC.PA": "Consumer Goods",  # LVMH Moët Hennessy Louis Vuitton SE
    "ATO.PA": "Technology",     # Atos SE
    "EN.PA": "Construction",    # Bouygues
    "RI": "Consumer Goods",     # Pernod Ricard SA
    "LR": "Manufacturing",      # Legrand SA
    "MT": "Manufacturing",      # ArcelorMittal
    "VIE": "Utilities",         # Veolia Environnement SA
    "SGO": "Manufacturing",     # Compagnie de Saint-Gobain SA
    "OR": "Consumer Goods",     # L'Oréal
    "ACKB": "Finance",               # Ackermans & Van Haaren
    "AGS": "Insurance",             # Ageas SA/NV
    "ABI": "Consumer Goods",        # Anheuser-Busch In Bev SA/NV
    "BPOST": "Transportation",      # Bpost S.A. / N.V
    "COFB": "Real Estate",          # Cofinimmo
    "COLR": "Retail",               # Colruyt Group N.V.
    "ELI": "Utilities",             # Elia Group
    "GLPG": "Healthcare",           # Galapagos NV
    "GBLB": "Investment",           # Groupe Bruxelles Lambert (New)
    "INGA": "Finance",              # ING Groep N.V.
    "KBC": "Finance",               # KBC Groupe NV
    "AD": "Retail",                 # Koninklijke Ahold Delhaize N.V.
    "BEKB": "Manufacturing",        # NV Bekaert SA.
    "ONTEX": "Consumer Goods",      # Ontex Group NV
    "PROX": "Telecommunications",   # Proximus SA
    "SOLB": "Chemicals",            # Solvay SA
    "TNET": "Telecommunications",   # Telenet Group Hldgs NV
    "UCB": "Healthcare",            # UCB SA
    "UMI": "Manufacturing",         # Umicore
    "AED": "Real Estate",           # Aedifica
    "APAM": "Manufacturing",        # Aperam
    "ARGX": "Healthcare",           # arGEN-X
    "MELE": "Technology",           # Melexis
    "SOF": "Finance",               # Sofina
    "WDP": "Real Estate" ,           # WDP
    "SAP": "Technology",             # SAP SE
    "SIE.DE": "Industrial",          # Siemens AG
    "AIR.PA": "Industrial",          # Airbus SE
    "DTE.DE": "Telecommunications",  # Deutsche Telekom AG
    "ALV.DE": "Finance",             # Allianz SE
    "P911.DE": "Automotive",         # Porsche Automobil Holding SE
    "MBG.DE": "Automotive",          # Mercedes-Benz Group AG
    "BMW.DE": "Automotive",          # Bayerische Motoren Werke AG
    "VOW3.DE": "Automotive",         # Volkswagen AG
    "MRK.DE": "Healthcare",          # Merck KGaA
    "SHL.DE": "Healthcare",          # Siemens Healthineers AG
    "DHL.DE": "Logistics",           # Deutsche Post AG
    "MUV2.DE": "Finance",            # Munich Re
    "IFX.DE": "Technology",          # Infineon Technologies AG
    "BAS.DE": "Chemicals",           # BASF SE
    "ADS.DE": "Consumer Goods",      # Adidas AG
    "EOAN.DE": "Utilities",          # E.ON SE
    "DB1.DE": "Finance",             # Deutsche Börse AG
    "BAYN.DE": "Healthcare",         # Bayer AG
    "RWE.DE": "Utilities",           # RWE AG
    "BEI.DE": "Consumer Goods",      # Beiersdorf AG
    "HEN3.DE": "Consumer Goods",     # Henkel AG & Co KGaA
    "DTG.F": "Industrial",           # Daimler Truck Holding AG
    "HNR1.DE": "Finance",            # Hannover Rück SE
    "DB": "Finance",                 # Deutsche Bank AG
    "VNA.DE": "Real Estate",         # Vonovia SE
    "SRT.DE": "Healthcare",          # Sartorius AG
    "HEI.DE": "Materials",           # HeidelbergCement AG
    "SY1.DE": "Chemicals",           # Symrise AG
    "CON.DE": "Automotive",          # Continental AG
    "CBK.F": "Finance",              # Commerzbank AG
    "QGEN": "Healthcare",            # Qiagen N.V.
    "1COV.F": "Chemicals",           # Covestro AG
    "ENR.F": "Energy",               # Siemens Energy AG
    "MTX.DE": "Aerospace",           # MTU Aero Engines AG
    "RHM.F": "Industrial",           # Rheinmetall AG
    "BNR.DE": "Chemicals",           # Brenntag AG
    "FME": "Healthcare",             # Fresenius Medical Care AG & Co KGaA
    "PAH3.DE": "Finance",            # Porsche Automobil Holding SE
    "FRE": "Healthcare",          # Fresenius SE & Co KGaA
    "RDSA.AS": "Energy",              # Royal Dutch Shell
    "ASRNL.AS": "Finance",            # ASR Nederland
    "NN.AS": "Finance",               # NN Group
    "GLPG.AS": "Healthcare",          # Galapagos
    "INGA.AS": "Finance",             # ING Groep
    "ADYEN.AS": "Technology",         # Adyen
    "DSM.AS": "Materials",            # DSM
    "ASM.AS": "Technology",           # ASM International
    "REN.AS": "Media",                # RELX
    "URW.AS": "Real Estate",          # Unibail-Rodamco-Westfield
    "AGN.AS": "Finance",              # Aegon
    "UNA.AS": "Consumer Goods",       # Unilever
    "KPN.AS": "Telecommunications",   # Koninklijke KPN
    "AD.AS": "Retail",                # Koninklijke Ahold Delhaize
    "PRX.AS": "Technology",           # Prosus
    "RAND.AS": "Services",            # Randstad
    "IMCD.AS": "Chemicals",           # IMCD
    "HEIA.AS": "Consumer Goods",      # Heineken
    "ASML.AS": "Technology",          # ASML Holding
    "ABN.AS": "Finance",              # ABN AMRO Bank
    "MT.AS": "Materials",             # ArcelorMittal
    "AKZA.AS": "Chemicals",           # Akzo Nobel
    "WKL.AS": "Professional Services",# Wolters Kluwer
    "PHIA.AS": "Healthcare",          # Koninklijke Philips
    "TKWY.AS": "Technology"           # Just Eat Takeaway.com
    # Note: Some tickers and domains might be repeated or may need updating based on latest company profiles
}

cac_40_tickers = [
    'TTE',     # TotalEnergies SE
    'AIR.PA',  # Airbus SE
    'SNY',     # Sanofi
    'SU.PA',   # Schneider Electric
    'AI.PA',   # Air Liquide
    'EL.PA',   # EssilorLuxottica
    'BNP.PA',  # BNP Paribas
    'SAF.PA',  # Safran
    'CS.PA',   # AXA
    'DG.PA',   # Vinci
    'STLA',    # Stellantis
    'DSY.PA',  # Dassault Systèmes
    'KER.PA',  # Kering
    'RI.PA',   # Pernod Ricard
    'STM',     # STMicroelectronics
    'ENGI.PA', # ENGIE
    'ACA.PA',  # Crédit Agricole
    'BN.PA',   # Danone
    'CAP.PA',  # Capgemini
    'SGO.PA',  # Compagnie de Saint-Gobain
    'ORAN',    # Orange
    'HO.PA',   # Thales
    'LR.PA',   # Legrand
    'ML.PA',   # Michelin
    'PUB.PA',  # Publicis Groupe
    'VIE.PA',  # Veolia
    'MT',      # ArcelorMittal
    'VIV.PA',  # Vivendi
    'AC.PA',   # Accor
    'SW.PA',   # Sodexo
    'CA.PA',   # Carrefour
    'GLE.PA',  # Société Générale
    'OR.PA',   # L'Oréal
    'RMS.PA',  # Hermès International
    'MC.PA',   # LVMH Moët Hennessy - Louis Vuitton
    'ATO.PA',  # Atos SE
    'EN.PA',   # Bouygues
   # 'DSY',      Dassault Systèmes SE
    #'RI',      # Pernod Ricard SA
    #'STLA'      Stellantis N.V.
]
bel_20_tickers = [
   # 'ACKB',  # Ackermans & Van Haaren
    'AGS',   # Ageas SA/NV
    #'ABI',   # Anheuser-Busch In Bev SA/NV
    #'BPOST', # Bpost S.A. / N.V
    #'COFB',  # Cofinimmo
    #'COLR',  # Colruyt Group N.V.
    #'ELI',   # Elia Group
    'GLPG',  # Galapagos NV
    #'GBLB',  # Groupe Bruxelles Lambert (New)
 #   'INGA',  # ING Groep N.V.
    'KBC',   # KBC Groupe NV
    #'AD',    # Koninklijke Ahold Delhaize N.V.
  #  'BEKB',  # NV Bekaert SA.
   # 'ONTEX', # Ontex Group NV
  #  'PROX',  # Proximus SA
  #  'SOLB',  # Solvay SA
    'TNET',  # Telenet Group Hldgs NV
    'UCB',   # UCB SA
    'UMI',   # Umicore
  #  'AED',   # Aedifica
    'APAM',  # Aperam
    'ARGX',  # arGEN-X
 #   'MELE',  # Melexis
  #  'SOF',   # Sofina
   # 'WDP'    # WDP
]
dax_tickers = [
    'SAP',      # SAP SE
    'SIE.DE',   # Siemens AG
    'AIR.PA',   # Airbus SE
    'DTE.DE',   # Deutsche Telekom AG
    'ALV.DE',   # Allianz SE
    'P911.DE',  # Porsche Automobil Holding SE
    'MBG.DE',   # Mercedes-Benz Group AG
    'BMW.DE',   # Bayerische Motoren Werke AG
    'VOW3.DE',  # Volkswagen AG
    'MRK.DE',   # Merck KGaA
    'SHL.DE',   # Siemens Healthineers AG
    'DHL.DE',   # Deutsche Post AG
    'MUV2.DE',  # Munich Re
    'IFX.DE',   # Infineon Technologies AG
    'BAS.DE',   # BASF SE
    'ADS.DE',   # Adidas AG
    'EOAN.DE',  # E.ON SE
    'DB1.DE',   # Deutsche Börse AG
    'BAYN.DE',  # Bayer AG
    'RWE.DE',   # RWE AG
    'BEI.DE',   # Beiersdorf AG
    'HEN3.DE',  # Henkel AG & Co KGaA
    'DTG.F',    # Daimler Truck Holding AG
    'HNR1.DE',  # Hannover Rück SE
    'DB',       # Deutsche Bank AG
    'VNA.DE',   # Vonovia SE
    'SRT.DE',   # Sartorius AG
    'HEI.DE',   # HeidelbergCement AG
    'SY1.DE',   # Symrise AG
    'CON.DE',   # Continental AG
    'CBK.F',    # Commerzbank AG
    'QGEN',     # Qiagen N.V.
    '1COV.F',   # Covestro AG
    'ENR.F',    # Siemens Energy AG
    'MTX.DE',   # MTU Aero Engines AG
    'RHM.F',    # Rheinmetall AG
    'BNR.DE',   # Brenntag AG
 #   'FME',      # Fresenius Medical Care AG & Co KGaA
    'PAH3.DE',  # Porsche Automobil Holding SE
  #  'FRE'       # Fresenius SE & Co KGaA
]
aex_tickers = [
  #  'RDSA.AS',   # Royal Dutch Shell
    'ASRNL.AS',  # ASR Nederland
    'NN.AS',     # NN Group
    'GLPG.AS',   # Galapagos
    'INGA.AS',   # ING Groep
    'ADYEN.AS',  # Adyen
  #  'DSM.AS',    # DSM
    'ASM.AS',    # ASM International
    'REN.AS',    # RELX
    #'URW.AS',    # Unibail-Rodamco-Westfield
    'AGN.AS',    # Aegon
    'UNA.AS',    # Unilever
    'KPN.AS',    # Koninklijke KPN
    'AD.AS',     # Koninklijke Ahold Delhaize
    'PRX.AS',    # Prosus
    'RAND.AS',   # Randstad
    'IMCD.AS',   # IMCD
    'HEIA.AS',   # Heineken
    'ASML.AS',   # ASML Holding
    'ABN.AS',    # ABN AMRO Bank
    'MT.AS',     # ArcelorMittal
    'AKZA.AS',   # Akzo Nobel
    'WKL.AS',    # Wolters Kluwer
    'PHIA.AS',   # Koninklijke Philips
    'TKWY.AS'    # Just Eat Takeaway.com
]

stock_symbols = ['AAPL', 'MSFT','AMZN','GOOG','NVDA'] + cac_40_tickers + bel_20_tickers + dax_tickers +aex_tickers

def download_and_save_stock_data(symbols, start_date, end_date, data_dir='data'):
    """
    Downloads and saves the stock and financial data for the given symbols from yfinance.
    Stores the data in the specified data directory with appropriate file names.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for symbol in symbols:
        # Download stock data
        stock_data_file = os.path.join(data_dir, f'{symbol}_stock_data.csv')
        if not os.path.isfile(stock_data_file):
            print(f"Downloading stock data for {symbol}")
            stock_data = yf.download(symbol, start=start_date, end=end_date)
            stock_data_diff=stock_data.pct_change().replace(np.inf,0).replace(-np.inf,0)
            stock_data_diff.to_csv(stock_data_file)
        else:
            print(f"Stock data for {symbol} already exists.")

        # Initialize ticker for financial data
        ticker = yf.Ticker(symbol)

        # Download financial data
        balance_file = os.path.join(data_dir, f'{symbol}_balance_data.csv')
        cash_file = os.path.join(data_dir, f'{symbol}_cash_data.csv')
        stmt_file = os.path.join(data_dir, f'{symbol}_stmt_data.csv')
        q_balance_file = os.path.join(data_dir, f'{symbol}_q_balance_data.csv')
        q_cash_file = os.path.join(data_dir, f'{symbol}_q_cash_data.csv')
        q_stmt_file = os.path.join(data_dir, f'{symbol}_q_stmt_data.csv')
        if os.path.isfile(balance_file):
            pass
        else:
            balance_sheet = ticker.balance_sheet.pct_change().replace(np.inf,0).replace(-np.inf,0)
            balance_sheet.to_csv(balance_file)

        #verufiy all three data sets are saved
        if os.path.isfile(cash_file):
            pass
        else:
            cash_flow = ticker.cashflow.pct_change().replace(np.inf,0).replace(-np.inf,0)
            cash_flow.to_csv(cash_file)
        if os.path.isfile(stmt_file):
            #date is the first row
            pass
        else:
            income_statement = ticker.income_stmt.pct_change().replace(np.inf,0).replace(-np.inf,0)
            income_statement.to_csv(stmt_file)
        if os.path.isfile(q_stmt_file):
            #date is the first row
            pass
        else:
            income_statement = ticker.quarterly_income_stmt.pct_change().replace(np.inf,0).replace(-np.inf,0)
            income_statement.to_csv(q_stmt_file)
        if os.path.isfile(q_balance_file):
            pass
        else:
            balance_sheet = ticker.quarterly_balance_sheet.pct_change().replace(np.inf,0).replace(-np.inf,0)
            balance_sheet.to_csv(q_balance_file)

        #verufiy all three data sets are saved
        if os.path.isfile(q_cash_file):
            pass
        else:
            cash_flow = ticker.quarterly_cashflow.pct_change().replace(np.inf,0).replace(-np.inf,0)
            cash_flow.to_csv(q_cash_file)


  # Add more stock symbols as needed
download_and_save_stock_data(stock_symbols, '2021-09-30', '2023-12-10')
def add_financial_feature(stock_data, financial_data, feature_name):
    financial_data.index = pd.to_datetime(financial_data.index)
    print(((financial_data.T.index == feature_name).any()),end=" ")
    if((financial_data.T.index == feature_name).any()):
        #print(financial_data.index)
    #    if feature_name in financial_data.columns:
    #        # Assuming that the financial data's index is DateTime
        feature_series = financial_data[feature_name].resample('D').ffill().reindex(stock_data.index, method='ffill')
        stock_data[feature_name] = feature_series
        return True
    else:
        stock_data[feature_name] = pd.DataFrame([0]*len(stock_data))
        return False

def add_moving_averages(df, periods):
    for n in periods:
        # Simple Moving Average
        df[f'SMA_{n}'] = df['Close'].rolling(window=n).mean()
        
        # Exponential Moving Average
        df[f'EMA_{n}'] = df['Close'].ewm(span=n, adjust=False).mean()

    return df
def read_and_prepare_data(symbols, data_dir='data'):
    """
    Reads the stock data and financial information for the given list of symbols,
    preprocesses it, and returns an aggregated DataFrame ready for training and testing.
    """
    aggregated_stock_data = []  # Initialize an empty DataFrame for aggregation

    for symbol in symbols:
        # Read stock data
        stock_data_file = os.path.join(data_dir, f'{symbol}_stock_data.csv')
        stock_data_ori = pd.read_csv(stock_data_file, index_col='Date', parse_dates=True)
        #print(symbol)
        stock_data = pd.DataFrame()
        stock_data.index = pd.to_datetime(stock_data_ori.index)
        stock_data["Close"]= stock_data_ori["Close"]
        stock_data["Volume"] = stock_data_ori["Volume"]
        periods = [5, 20, 50]  # You can choose different periods as per your need
        stock_data = add_moving_averages(stock_data, periods)
                # Ensure the stock_data index is a DatetimeIndex
        #stock_data.index = pd.to_datetime(stock_data.index)
        # Get financial data
        balance_file = os.path.join(data_dir, f'{symbol}_balance_data.csv')
        cash_file = os.path.join(data_dir, f'{symbol}_cash_data.csv')
        stmt_file = os.path.join(data_dir, f'{symbol}_stmt_data.csv')
        q_balance_file = os.path.join(data_dir, f'{symbol}_q_balance_data.csv')
        q_cash_file = os.path.join(data_dir, f'{symbol}_q_cash_data.csv')
        q_stmt_file = os.path.join(data_dir, f'{symbol}_q_stmt_data.csv')
        balance_sheet = pd.read_csv(balance_file, parse_dates=True, index_col=0)
        cash_flow = pd.read_csv(cash_file, parse_dates=True, index_col=0)
        income_statement = pd.read_csv(stmt_file, parse_dates=True, index_col=0)
        q_balance= pd.read_csv(q_balance_file, parse_dates=True, index_col=0)
        q_cash= pd.read_csv(q_cash_file, parse_dates=True, index_col=0)
        q_stmt= pd.read_csv(q_stmt_file, parse_dates=True, index_col=0)
        # Transpose financial data
        balance_sheet = balance_sheet.T
        cash_flow = cash_flow.T
        income_statement = income_statement.T
        q_balance = q_balance.T
        q_cash = q_cash.T
        q_stmt = q_stmt.T
        #print(balance_sheet.index)
        print("\n"+symbol,end=" ")
        # Example: Incorporate selected financial data into stock_data
        
        financial_features = [ 'Net PPE','Cash Financial','Ordinary Shares Number', 'Tangible Book Value', 'Invested Capital', 'Net Tangible Assets', 'Total Assets', 'Total Debt', 'Capital Stock', 'Common Stock Equity', 'Net PPE']
        """
                # Ensure the stock_data index is a DatetimeIndex
        balance_sheet.index = pd.to_datetime(balance_sheet.index)

        for feature in financial_features:
            # Assuming the financial data's index is DateTime
            add_financial_feature(stock_data,balance_sheet,feature)
        #_q_balance_data
        # Incorporate other financial features like Free Cash Flow, Net Income, Total Revenue, and Gross Profit
        add_financial_feature(stock_data, cash_flow, 'Free Cash Flow')
        add_financial_feature(stock_data, cash_flow, 'Cash Dividends Paid')
        for feat in ['Net Income','Total Revenue','Gross Profit','Net Income From Continuing And Discontinued Operation']:
            add_financial_feature(stock_data, income_statement, feat)   
        """    
        """ 
        count=0
        for feature in financial_features:
            # Assuming the financial data's index is DateTime
            if(add_financial_feature(stock_data,q_balance,feature)):
                count=count+1
        #_q_balance_data
        # Incorporate other financial features like Free Cash Flow, Net Income, Total Revenue, and Gross Profit
        for e in ['Free Cash Flow', 'Operating Cash Flow', 'Change In Working Capital', 'Net Income From Continuing Operations', 'Financing Cash Flow', 'Investing Cash Flow', 'End Cash Position']:
            if(add_financial_feature(stock_data, q_cash, e)):
                count = count+1
        for feat in ['Tax Effect Of Unusual Items', 'Normalized Income', 'Diluted Average Shares', 'Diluted EPS', 'Diluted NI Availto Com Stockholders', 'Tax Provision', 'Pretax Income', 'Total Revenue', 'Net Income From Continuing Operation Net Minority Interest', 'Net Interest Income',"Operating Expense"]:
            if(add_financial_feature(stock_data, q_stmt, feat)):
                count = count+1
        """
        domain_value=domain_of_activity.get(companies_domains.get(symbol))
        stock_data["domain"] = pd.DataFrame([domain_value]*len(stock_data))
        # Fill missing values
        stock_data.fillna(0, inplace=True)

        # Ensure the stock_data index is a DatetimeIndex
        stock_data.index = pd.to_datetime(stock_data.index)

        # Concatenate or append the stock data to the aggregated DataFrame
        if(True):#count>23
            aggregated_stock_data.append(stock_data)

    return aggregated_stock_data



# Fill missing values with -1
#stock_data.fillna(0, inplace=True)
"""
# Preprocess data
features = stock_data.drop('Close', axis=1).values
target = stock_data['Close'].values

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
target_scaled = scaler.fit_transform(target.reshape(-1, 1)).ravel()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target_scaled, test_size=0.2, random_state=42)
"""
# Preprocess data for predicting next day's closing price
number_days = 30  # Number of days to consider for prediction
stock_data = read_and_prepare_data(stock_symbols)
# Function to create a dataset for time series prediction
def create_time_series_dataset(data, n):
    X, y = [], []
    for i in range(len(data) - n):
        X.append(data.iloc[i:(i + n), :-1].values)
        y.append(data.iloc[i + n, -1])
    return np.array(X), np.array(y)
def preprocess_data_for_time_series(stock_datas, number_days=30):
    """
    Processes the provided stock_data to create a time series dataset.
    Scales the features and target and splits the data into training and test sets.

    Parameters:
    stock_data (DataFrame): The stock data to be processed.
    number_days (int): The number of days to consider for prediction.

    Returns:
    X_train, X_test, y_train, y_test: Training and testing datasets.
    """
    signal = 0
    X_train, X_test, y_train, y_test = np.array([]),np.array([]),np.array([]),np.array([])
    for stock in stock_datas:
        # Shift the 'Close' column to get the next day's closing price as the target
        #print(stock['Close'].shift(-1))
        stock['Next Close'] = stock['Close'].shift(-1)
        stock.dropna(inplace=True)  # Drop the last row with NaN target



        # Save processed stock data to a file
        stock.to_csv('stock_data_final.csv')

        # Create time series dataset
        #X, y = create_time_series_dataset(stock, number_days)
        if signal == 0 :
            X, y = create_time_series_dataset(stock, number_days)
            signal = 1
            
        else:
            temp1,temp2= create_time_series_dataset(stock, number_days)
            #print(type(X))
            #print(type(y))
            #X = X+temp1
            #y =y+temp2
            X =np.concatenate((X,temp1))
            y = np.concatenate((y,temp2))
            
    #print(X[0])
    #X.to_csv('stock_data_final.csv')
    # Scale the features and target
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    #print(X_train)
    # Split the data

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)


    return X_train, X_test, y_train, y_test, scaler_X,scaler_y
X_train, X_test, y_train, y_test, scaler_X,scaler_y = preprocess_data_for_time_series(stock_data, number_days)
print(X_train[0], y_train[0])
# Example usage for a single symbol


#X_train, X_test, y_train, y_test = create_time_series_dataset(data, number_days)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

def initialize_models():
    """
    Initializes and returns a dictionary of machine learning models.
    """
    models = {
        'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'LinearRegression': LinearRegression(),
        'SVR': SVR()
    }
    return models

# Initialize models
models = initialize_models()
from sklearn.metrics import mean_squared_error

def evaluate_models(models, X_train, y_train, X_test, y_test, scaler_X,scaler_y):
    """
    Trains and evaluates the given models using the provided train and test sets.
    Prints out the Mean Squared Error for each model.
    """
    for name, model in models.items():
        model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        
        # Predict on the test set
        predictions = model.predict(X_test.reshape(X_test.shape[0], -1))

        # Inverse transform predictions and true values to original scale
        predictions_original = scaler_y.inverse_transform(predictions.reshape(-1, 1)).ravel()
        y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

        # Calculate MSE in original dollar scale
        mse_dollars = mean_squared_error(y_test_original, predictions_original)
        print(f"{name} - Mean Squared Error in dollars: ${mse_dollars:.2f}")
#print(len(X_train))
# Example usage (you need to define X_train, y_train, X_test, y_test)
#evaluate_models(models, X_train, y_train, X_test, y_test, scaler_X,scaler_y)
# Reshape data for LSTM model (required 3D shape)
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights

batch_size = 32

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_freq=5*batch_size
                                                 )

X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
X_test_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
# Build LSTM model
hyperparams = {
    'lstm_units': [[250, 200, 200], [300, 250, 250]],
    'dense_units': [[50, 1], [100, 1]],
    'batch_size': [32],
    'epochs': [10],
    'learning_rate': [0.001]  # Example learning rates
}
best_mse = float('inf')
best_params = {}
for lstm_units in hyperparams['lstm_units']:
    for dense_units in hyperparams['dense_units']:
        for batch_size in hyperparams['batch_size']:
            for epochs in hyperparams['epochs']:
                for lr in hyperparams['learning_rate']:
                    # Build and train the model
                    model = build_lstm_model(X_train_lstm.shape[1:], lstm_units, dense_units, lr)
                    model.fit(X_train_lstm, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test_lstm, y_test_original))

                    # Predict and evaluate
                    predictions = model.predict(X_test_lstm)
                    predictions_original = scaler_y.inverse_transform(predictions).ravel()
                    mse = mean_squared_error(y_test_original, predictions_original)

                    if mse < best_mse:
                        best_mse = mse
                        best_params = {
                            'lstm_units': lstm_units,
                            'dense_units': dense_units,
                            'batch_size': batch_size,
                            'epochs': epochs,
                            'learning_rate': lr
                        }

                    # Log results
                    with open("hyperparam_tuning_log.txt", "a") as f:
                        f.write(f"Params: {lstm_units, dense_units, batch_size, epochs, lr}, MSE: {mse}\n")

print(f"Best MSE: {best_mse} with params: {best_params}")
#https://deepai.org/machine-learning-glossary-and-terms/epoch
