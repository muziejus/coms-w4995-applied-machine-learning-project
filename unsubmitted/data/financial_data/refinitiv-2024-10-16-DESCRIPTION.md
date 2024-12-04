# Description of Refinitiv Query 2024-10-16

On November 5, 2024, we ran this query on Refinitiv:

file	:	ds2primqtprc
qvar	:	ticker
beg_d	:	01
beg_m	:	Jan
datef	:	YYMMDDd10.
end_d	:	14
end_m	:	Oct
exchg	:	0
idvar	:	infocode
query	:	LULU WBA ULTA DLTR WMT
beg_yr	:	2019
end_yr	:	2024

format	:	csv
keyvar	:	infocode
method	:	1
qvards	:	wrds_ds_names
datevar	:	marketdate
library	:	tfn
compress	:	
id_table	:	wrds_ds_names
wrdsversion	:	3
file_to_upload	:	
saved_query_name	:	Naive Five Company Price Search
query_form_page_id	:	5914
source_saved_query	:	
data_dictionary_table	:	ds2primqtprc
query_form_is_preview	:	false
query_form_environment	:	[WRDS Prod wrds-pub3-w] 
data_dictionary_product	:	tr_ds_equities
query_form_page_revision	:	69251

The query was for price information on:
- LULU (Lululemon)
- WMT (Walmart)
- WBA (Walgreens)
- ULTA (Ulta)
- DLTR (Dollar Tree)

Below are the various variables in this download.

## Variable Reference

| Variable Name | Type | Description|
| --------------|------|--------------|
| infocode | Decimal | InfoCode - Primary mapping code for Country-level Identifiers (infocode) |
| dscode | Char | DsCode - Datastream Code (dscode) |
| dsseccode | Decimal | DsSecCode - Quantitative Analytics internal security code (dsseccode) |
| region | Char | Region - Region or Country code indicating where the security is traded (region) |
| regcodetypeid | Int | RegCodeTypeId - Region type code identifier (regcodetypeid) |
| isprimqt | Int | IsPrimQt - flags whether this is the primary country-level quote (isprimqt) |
| dsqtname | Char | DsQtName - Name of the name of country-level quote (dsqtname) |
| dslocalcode | Char | DsLocalCode - Country code combined with the exchange ticker (dslocalcode) |
| dsmnem | Char | DsMnem - Datastream mnemonic for the equity (dsmnem) |
| covergcode | Char | CovergCode - Indicates the type of coverage (covergcode) |
| statuscode | Char | StatusCode - Status code indicating the status of the item (statuscode) |
| permid | Char | PermID - Permanent country-level ID associated with the item (permid) |
| typecode | Char | TypeCode - Code indicating the type of equity (typecode) |
| primisocurrcode | Char | PrimISOCurrCode - Three-character primary ISO currency code of the security’s country of origin (primisocurrcode) |
| delistdate | Date | DelistDate - Delisted date. This field is no longer available in this table (delistdate) |
| dssctycode | Char | DsSctyCode - Datastream security code (dssctycode) |
| dscmpycode | Decimal | DsCmpyCode - Quantitative Analytics internal company code (dscmpycode) |
| ismajorsec | Char | IsMajorSec - Indicates whether this security is the company's primary security (ismajorsec) |
| dssecname | Char | DsSecName - Security name (dssecname) |
| isocurrcode | Char | ISOCurrCode - Code for the currency of the country of origin of the security (isocurrcode) |
| divunit | Char | DivUnit - Dividend units (divunit) |
| primqtsedol | Char | PrimQtSedol - SEDOL of the security's primary quote (primqtsedol) |
| primexchmnem | Char | PrimExchMnem - Primary exchange code of the security (primexchmnem) |
| primqtinfocode | Decimal | PrimQtInfoCode - Quantitative Analytics internal InfoCode of a security's primary quote (primqtinfocode) |
| wssctyppi | Char | WSSctyPPI - Worldscope Security ID (wssctyppi) |
| ibesticker | Char | IBESTicker - I/B/E/S ticker (ibesticker) |
| wssctyppi2 | Char | WSSctyPPI2 - Secondary Worldscope Security ID (wssctyppi2) |
| ibesticker2 | Char | IBESTicker2 - Secondary I/B/E/S ticker (ibesticker2) |
| security_delistdate | Date | DelistDate - Delisted date (security_delistdate) |
| dscompcode | Char | DsCompCode - Datastream Company Code (dscompcode) |
| dscmpyname | Char | DsCmpyName - Company name (dscmpyname) |
| cmpyctrycode | Char | CmpyCtryCode - Country where the Company's main activities are (cmpyctrycode) |
| cmpyctrytype | Int | CmpyCtryType - Company Country Type (cmpyctrytype) |
| indusisdef | Char | IndusIsDef - Industry Code Indicator (indusisdef) |
| startdate | Date | StartDate - Datastream change date (startdate) |
| enddate | Date | EndDate - Last date the ISIN was valid (enddate) |
| isin | Char | ISIN - Period between the specified StartDate and EndDate (isin) |
| isin2 | Char | ISIN2 - Period between the specified StartDate and EndDate. Applicable for Thailand and Malaysian (isin2) |
| licflagc | Int | LicFlagC - CUSIP-based ISINs (licflagc) |
| ticker | Char | ticker |
| infocode | Decimal | InfoCode - Quantitative Analytics internal code for Datastream tables (infocode) |
| marketdate | Date | MarketDate - Date of the price (marketdate) |
| exchintcode | Int | ExchIntCode - Integer code for the exchange (exchintcode) |
| isocurrcode | Char | ISOCurrCode - ISO Currency Code for the currency upon which the series is calculated (isocurrcode) |
| refprctypcode | Int | RefPrcTypCode - Identifies whether the price is final (refprctypcode) |
| open_ | Float | Open_ - Opening price for a security on a given date (open_) |
| high | Float | High - High price for the security on a given date or date range (high) |
| low | Float | Low - Low price for the security on a given date or date range (low) |
| close_ | Float | Close_ - Closing price of the security on the given date (close_) |
| volume | Float | Volume - Unadjusted volume (volume) |
| bid | Float | Bid - Bid price for the security on the given date (bid) |
| ask | Float | Ask - Ask price for the security on the given date (ask) |
| vwap | Float | VWAP - Volume-weighted average price (vwap) |
| mosttrdprc | Float | MostTrdPrc - Traded price (mosttrdprc) |
| consolvol | Float | ConsolVol - Consolidated volume of the security (consolvol) |
| mosttrdvol | Float | MostTrdVol - Traded volume (mosttrdvol) |
| licflag | Int | LicFlag is the bitmap license flag (licflag) |


