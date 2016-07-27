train =  pd.read_csv("data/Train.csv")

train_clean = train.drop(['SalesID', 'MachineID', 'ModelID', 'UsageBand', 'fiModelDesc', 'fiBaseModel', 'fiSecondaryDesc', 'fiModelSeries', 'fiModelDescriptor', 'fiProductClassDesc', 'ProductGroup', 'ProductGroupDesc','Drive_System', 'Forks', 'Pad_Type', 'Ride_Control', 'Stick', 'Transmission', 'Turbocharged', 'Blade_Width', 'Enclosure_Type', 'Engine_Horsepower', 'Pushblock', 'Ripper', 'Scarifier', 'Tip_Control', 'Tire_Size', 'Coupler', 'Coupler_System', 'Grouser_Tracks', 'Hydraulics_Flow', 'Undercarriage_Pad_Width', 'Undercarriage_Pad_Width', 'Stick_Length', 'Thumb', 'Pattern_Changer', 'Grouser_Type', 'Backhoe_Mounting', 'Blade_Type', 'Travel_Controls', 'Differential_Type', 'Steering_Controls'], axis = 1)

#Datasource dummies:
datasource_dum = pd.get_dummies(train_clean['datasource'])
datasource_dum = datasource_dum.drop([121], axis = 1)
train_clean[['datasource132', 'datasource136', 'datasource149', 'datasource172']] = datasource_dum
train_clean = train_clean.drop(['datasource'], axis = 1)

#Filling auctioneerID NA's with -1, and getting dummy variables.
train_clean['auctioneerID'] = train_clean['auctioneerID'].fillna(-1)
auctioneerID_dum = pd.get_dummies(train_clean['auctioneerID'])
train_clean[['aucID_-1', 'aucID_0', 'aucID_1', 'aucID_2', 'aucID_3', 'aucID_4', 'aucID_5', 'aucID_6', 'aucID_7', 'aucID_8', 'aucID_9', 'aucID_10', 'aucID_11', 'aucID_12', 'aucID_13', 'aucID_14', 'aucID_15', 'aucID_16', 'aucID_17', 'aucID_18', 'aucID_19', 'aucID_20', 'aucID_21', 'aucID_22', 'aucID_23', 'aucID_24', 'aucID_25', 'aucID_26', 'aucID_27', 'aucID_28', 'aucID_99']] = auctioneerID_dum

#Creating a Vintage variable for equipment made prior to 1970
train_clean['Vintage'] = (train_clean['YearMade'] < 1970).astype(int)

#Filling Machine Hour NA's with median value of non-zero's. Median = 3138
notnullmask = pd.notnull(train_clean['MachineHoursCurrentMeter']) #Removing nulls.
not_nulls = train_clean[notnullmask]
MachineHourMedian = (not_nulls[not_nulls['MachineHoursCurrentMeter']!=0]['MachineHoursCurrentMeter']).median()
train_clean['MachineHoursCurrentMeter'] = train_clean['MachineHoursCurrentMeter'].fillna(MachineHourMedian)

#Dummies for product size
prod_size_dum = pd.get_dummies(train_clean['ProductSize'])
train_clean[['ps_compact', 'ps_large', 'ps_large/medium', 'ps_medium', 'ps_mini', 'ps_small']] = prod_size_dum

#Dummies for states.
state_dum = pd.get_dummies(train_clean['state'])
train_clean[['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California',
       'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia',
       'Hawaii', 'Idaho', 'Illinois', 'Indiana', u'Iowa', 'Kansas',
       'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts',
       'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana',
       'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico',
       'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma',
       'Oregon', 'Pennsylvania', 'Puerto Rico', 'Rhode Island',
       'South Carolina', 'South Dakota', 'Tennessee', 'Texas',
       'Unspecified', 'Utah', u'Vermont', 'Virginia', 'Washington',
       'Washington DC', 'West Virginia', 'Wisconsin', 'Wyoming']] = state_dum
train_clean = train_clean.drop(['Unspecified'], axis = 1)

#Dummies for Enclosure
enclosure_dum = pd.get_dummies(train_clean['Enclosure'])

#Dummies for blade extension
blade_extension_dum = pd.get_dummies(train_clean['Blade_Extension'])

#Dummies for Hydraulics
hydraulics_dum = pd.get_dummies(train_clean['Hydraulics'])

#Dummies for Track_type
track_type_dum = pd.get_dummies(train_clean['Track_Type'])

#Anything before year 1970 is vintage. Anything older will have the number of years after
#1970 that it was built.

year_lambda = lambda x: x-1960
from1960 = year_lambda(train_clean['YearMade']).map(lambda x: 0 if x <0 else x)
def yearbin(x):
    if x < 1900:
        return "None"
    if x > 1899 and x < 1910:
        return 1900
    elif x >= 1910 and x < 1920:
        return 1910
    elif x >= 1920 and x < 1930:
        return 1920
    elif x >= 1930 and x < 1940:
        return 1930
    elif x >= 1940 and x < 1950:
        return 1940
    elif x >= 1950 and x < 1960:
        return 1950
    elif x >= 1960 and x < 1970:
        return 1960
    elif x >= 1970 and x < 1980:
        return 1970
    elif x >= 1980 and x < 1990:
        return 1980
    elif x >= 1990 and x < 2000:
        return 1990
    elif x >= 2000 and x < 2010:
        return 2000
    elif x >= 2010 and x < 2020:
        return 2010

yearbins = train_clean['YearMade'].apply(yearbin)
train_clean['YearBin'] = yearbins


'''
Baseline info:
datasource = 121
auctioneerID = Nan
ProductSize = Nan
State = unspecified

'''
