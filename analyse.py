import os
import pandas as pd

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
import os
import pandas as pd

from collections import Counter
def get_most_encountered_fields(data_dir, symbols, file_types):
    field_counts = {file_type: {} for file_type in file_types}

    for symbol in symbols:
        for file_type in file_types:
            file_path = os.path.join(data_dir, f"{symbol}_{file_type}.csv")
            try:
                df = pd.read_csv(file_path)
                print(df.T.columns)
                for col in df.T.columns:
                    if col in field_counts[file_type]:
                        field_counts[file_type][col] += 1
                    else:
                        field_counts[file_type][col] = 1
            except FileNotFoundError:
                print(f"File not found: {file_path}")
                continue

    most_encountered_fields = {}
    for file_type, counts in field_counts.items():
        if counts:  # Check if there are counted fields
            most_encountered_fields[file_type] = max(counts, key=counts.get)

    return most_encountered_fields

def analyze_symbols(data_dir, symbols, file_types):
    symbol_common_fields = {}
    for symbol in symbols:
        symbol_files = [os.path.join(data_dir, f"{symbol}_{file_type}.csv") for file_type in file_types]
        common_fields = get_common_fields(symbol_files)
        symbol_common_fields[symbol] = common_fields
    return symbol_common_fields
file_types = ['q_balance_data', 'q_cash_data', 'q_stmt_data']
def get_most_common_rows(data_dir, symbols, file_types):
    row_counts = {file_type: Counter() for file_type in file_types}

    for symbol in symbols:
        for file_type in file_types:
            file_path = os.path.join(data_dir, f"{symbol}_{file_type}.csv")
            try:
                df = pd.read_csv(file_path)
                for row in df.itertuples(index=False, name=None):  # Convert rows to tuples
                    row_counts[file_type][row] += 1
            except FileNotFoundError:
                print(f"File not found: {file_path}")
                continue

    most_common_rows = {}
    for file_type, counter in row_counts.items():
        if counter:  # Check if there are counted rows
            most_common_row, count = counter.most_common(1)[0]  # Get the most common row
            most_common_rows[file_type] = (most_common_row, count)

    return most_common_rows
def get_top_common_rows(data_dir, symbols, file_types, top_n=10):
    row_counts = {file_type: Counter() for file_type in file_types}

    for symbol in symbols:
        for file_type in file_types:
            file_path = os.path.join(data_dir, f"{symbol}_{file_type}.csv")
            try:
                df = pd.read_csv(file_path)
                for row in df.itertuples(index=False, name=None):  # Convert rows to tuples
                    row_counts[file_type][row] += 1
            except FileNotFoundError:
                print(f"File not found: {file_path}")
                continue

    top_common_rows = {}
    for file_type, counter in row_counts.items():
        if counter:  # Check if there are counted rows
            top_rows = counter.most_common(top_n)  # Get the top N common rows
            top_common_rows[file_type] = top_rows

    return top_common_rows

def get_top_common_nonzero_rows(data_dir, symbols, file_types, top_n=10):
    row_counts = {file_type: Counter() for file_type in file_types}

    for symbol in symbols:
        for file_type in file_types:
            file_path = os.path.join(data_dir, f"{symbol}_{file_type}.csv")
            try:
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    if row.iloc[1:].astype(bool).any():  # Check if any non-zero value after the first column
                        row_name = row.iloc[0]  # Assuming the row name is in the first column
                        row_counts[file_type][row_name] += 1
            except FileNotFoundError:
                print(f"File not found: {file_path}")
                continue

    top_common_rows = {}
    for file_type, counter in row_counts.items():
        if counter:  # Check if there are counted rows
            top_rows = counter.most_common(top_n)  # Get the top N common rows
            top_common_rows[file_type] = top_rows

    return top_common_rows

# Example usage
data_dir = 'data'
symbols = ['symbol1', 'symbol2', 'symbol3', ..., 'symbol80']  # Replace with your symbols




top_common_rows = get_top_common_nonzero_rows(data_dir, stock_symbols, file_types)
for file_type, rows in top_common_rows.items():
    print(f"Top 10 common non-zero rows in '{file_type}':")
    for row_name, count in rows:
        print(f"  Row Name: {row_name}, Count: {count}")

for file_type, rows in top_common_rows.items():
    print(f"{file_type}: {[row_name for row_name, _ in rows]}")
