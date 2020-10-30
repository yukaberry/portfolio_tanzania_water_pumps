import pandas as pd
import numpy as np



def installer_cl(df):
    """keep top 10 of installer"""

    if df['installer']=="DWE":
        return 'dwe'
    elif df['installer']=="Government":
        return 'gov'
    elif df['installer']=="RWE":
        return "rwe"
    elif df['installer']=="Commu":
        return"commu"
    elif df['installer']=="DANIDA":
        return"danida"
    elif df['installer']=="KKKT":
        return "kkkt"
    elif df['installer']=="Hesawa":
        return "hesewa"
    elif df['installer']=="0":
        return "unknown"
    elif df['installer']=="TCRS":
        return"tcrs"
    elif df['installer']=="Central government":
        return "central gov" 
    elif df['installer']=="NaN":
        return "nan"
    else:
        return "others"


def funder_cl(df):
    """keep top 10 of installer"""

    if df['funder']=="Government Of Tanzania":
        return 'gov of Tanzania'
    elif df['funder']=="Danida":
        return 'danida'
    elif df["funder"]=="Hesawa":
        return "hesawa"
    elif df['funder']=="Rwssp":
        return"rwssp"
    elif df['funder']=="World Vision":
        return"world vision"
    elif df['funder']=="Unicef":
        return "unicef"
    elif df['funder']=="Hesawa":
        return "hesewa"
    elif df['funder']=="Tasaf":
        return "tasaf"
    elif df['funder']=="District Council":
        return"district council"
    elif df['funder']=="Kkkt":
        return "kkkt" 
    elif df['funder']=="NaN":
        return "nan"
    elif df["funder"]=="0":
        return "unknown"
    else:
        return "others"


def year_cl(df):
    """Construction year grouping"""

    if df["construction_year"]==0:
        return 'unknown'
    elif 1960 <= df["construction_year"]<=1969:
        return "60s"
    elif 1970 <= df["construction_year"]<=1979:
        return "70s"
    elif 1980 <= df["construction_year"]<=1989:
        return "80s"
    elif 1990 <= df["construction_year"]<=1999:
        return "90s"
    elif 2000 <= df["construction_year"]<=2009:
        return "00s"
    elif 2010 <= df["construction_year"]<=2019:
        return "10s"



def pop_cl(df):
    """Population zero or not zero"""

    if df["population"]==0:
        return "1"
    else:
        return "0"


def pay_cl(df):
    """payment never pay , pay or unknown"""

    if df["payment"]=="never pay":
        return "NeverPay"
    elif df["payment"]=="unknown":
        return "Unknown"
    else:
        return "Pay"






