import pandas import pd

# Adjusts the price of a single sale for inflation to 2016 dollars
def adj_sale(row):
    price_that_year = float(inflation[inflation.YEAR == row.saledate.year].PRICE)
    return row.SalePrice*pres_val/price_that_year

# Returns a dataframe with a new column of prices adjusted to 2016 dollars
def clean_inflation_adjustment(df):
    inflation = pd.read_csv('inflation.csv')
    inflation[inflation.YEAR == df.saledate[0].year].INFLATION
    pres_val = float(inflation[inflation.YEAR == 2015].PRICE)
    df['adj_sale_price'] = df.apply(adj_for_inflation, axis=1)
    return df

if __name__ == "__main__":
    df = pd.read_csv('Train.csv')
    df.saledate = df.saledate.apply(lambda x: datetime.strptime(str(x), '%m/%d/%Y %H:%M'))
    clean_inflation_adjustment(df)
