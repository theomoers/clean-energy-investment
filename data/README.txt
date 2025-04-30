
===============================================================
Clean Investment Monitor Data Download: 2018-Q1 through 2024-Q4
===============================================================

The Clean Investment Monitor (CIM) is a joint project of Rhodium Group
and MIT’s Center for Energy and Environmental Policy Research (CEEPR) 
that tracks investments in the manufacture and deployment of covered 
technologies that reduce GHG emissions in the United States. All 
analysis is made publicly available at cleaninvestmentmonitor.org. 

This folder includes underlying data used by the dashboards and the
report, and provides additional facility-level details for manufacturing, 
utility-scale energy, and industrial facilities.

See also the latest report and full documentation at
cleaninvestmentmonitor.org.

The “Clean Investment Monitor” data on public and private investments
in the manufacturing and deployment of emission-reducing technologies,
including all charts and depictions of such data, is licensed under a 
Creative Commons Attribution 4.0 International License (CC BY 4.0).
See https://creativecommons.org/licenses/by/4.0/ for the license text.

release sub-version: 2024_Q4.20250218.0
updated: Tue Feb 18 16:00:32 2025 (MST)

DOLLAR YEAR: 2023
INVESTMENT UNITS: 2023 $Million

File & Column Descriptions
==========================

quarterly_actual_investment.csv
-------------------------------
Total estimated actual investments by state, technology, and quarter.

Column descriptions:
 * Segment
     Market segment. One of "Manufacturing", "Energy and Industry", or
     "Retail." "Manufacturing" indicates an investment in facilities or 
     capacity to produce GHG-reducing technology. "Energy and Industry" 
     refers to the deployment of technologies that reduce GHG emissions 
     in the bulk production of energy or industrial goods or that 
     capture ambient carbon dioxide. "Retail" refers to the purchase and
     installation of technology by individual households and businesses.
 * State
     US State in which the investment was made
 * Technology
     Broad technology group. See the Technology and Subcategory
     Descriptions section at the end of the readme.
 * Subcategory
     Technology detail group. This field provides additional detail
     about the type of activity occurring at the facility.
 * quarter
     The quarter in which the investment is estimated to take place in
     the format YYYY-QQ
 * Estimated_Actual_Quarterly_Expenditure
     Estimated quarterly investment, reported in 2023 Million USD.
     The time series of actual investment is based on reported investment totals, 
     extrapolated based on reported capacity and facility characteristics,
     estimated by multiplying retail sales by average prices or MSRP,
     or modeled based on known investments from similar facilities.
     Manufacturing, utility-scale energy, and industrial investments are 
     spread evenly over the duration of each facility's construction period.
     Construction start and end dates may be reported (e.g. in monthly 
     status filings, news sources or press releases) or may be estimated
     based on typical construction times for similar projects. Retail
     investments are entirely attributed to the quarter in which the sale
     or interconnection was reported.
* Decarb_Sector
     End use sector where the deployed technology contributes to emissions 
     reductions. See methodology for detailed mapping.

manufacturing_energy_and_industry_facility_metadata.csv
-------------------------------------------------------
Detailed metadata for manufacturing, utility-scale energy, and industrial facilities

The data in this file represent a detailed breakout for the Manufacturing and 
Energy & Industry segments; all facilities here which have broken ground during 
the 2018-Q1 through 2024-Q4 window 
are included in `quarterly_actual_investment.csv`, aggregated to the 
state-by-technology level. The estimated total facility capital expenditures 
here are announced investment values. 

Note that investments corresponding to electricity generators < 1MW are reported
at the state level in the Retail Distributed Electricity and Storage sections of
the `quarterly_actual_investment.csv` file and are not included here, because data 
detailing individual distributed generation resources are not available in all states. 


Column descriptions:
 * unique_id
     CIM-specific unique identifier for the facility. Use this to track
     facilities across release versions.
 * Segment
     Market segment. One of "Manufacturing" or "Energy and Industry".
     "Manufacturing" indicates an investment in facilities or capacity
     to produce GHG-reducing technology. "Energy and Industry" refers
     to the deployment of technologies that reduce GHG emissions in 
     the bulk production of energy or industrial goods or that capture
     ambient carbon dioxide.
 * Company
     Company or facility name
 * Technology
     Broad technology group. See the Technology and Subcategory
     Descriptions section below.
 * Subcategory
     Technology detail group. This field provides additional detail
     about the type of activity occurring at the facility.
 * Decarb_Sector
     End use sector where the deployed technology contributes to emissions 
     reductions. See methodology for detailed mapping.
 * Project_Type
     New: Investment in the construction of a new facility
     Expansion: Investment in new capacity at an existing facility
     Restart: Investments associated with the re-opening of
         a retired or offline facility
     Retrofit: Facilities with investments enabling the production
         of different or improved products
     Canceled: Planned projects which have been canceled or
         permanently delayed
 * Current_Facility_Status 
     A / Announced:
        Includes Permitting and Design phase. Does not include 
        announcements of "intent" without specifying a specific location
        or beginning Front-End Engineering & Design work. Projects which
        have announced an intention to construct a facility but have not
        yet met these criteria are not included in the CIM dataset.
     U / Under Construction:
        Under construction, or post-construction but not yet operating
     O / Operating:
         Operating or offline but planned to return to operation
     C / Canceled prior to operation:
         Canceled or development/construction on indefinite hold
     R / Retired:
         Retired or offline with no plans to return to operation
 * Announcement_Date
     Date the project was first announced. If not available, may be the
     publication date of a news article or press release covering the
     announcement. In the case of utility-scale electricity generation
     technologies, this is the first day of the initial month in which the
     facility appears in an EIA-860M filing. In some cases, this date is
     estimated based on planning and construction durations for similar
     facilities when a facility is first announced as operational and no
     information about the announcement or construction period can be
     found.
 * Total_Facility_CAPEX_Estimated
     Estimated total investment required to construct the proposed or actual 
     facility, converted to 2023 Million USD, based on reported 
     investment, extrapolated based on reported capacity and facility 
     characteristics, or modeled based on known investments from similar 
     facilities. Note that this total investment may take place before, 
     during, and/or after the CIM time horizon, including in future quarters,
     if facility construction/installation is not yet complete. This
     investment total is also reported for canceled projects; therefore, the
     total investment reported here may never be fully realized.
 * State
     US State in which the investment was made
 * Address
     Address or approximate location of the facility (if available)
 * Latitude
     Reported, estimated, or approximate latitude of the facility (if available)
 * Longitude
     Reported, estimated, or approximate longitude of the facility (if available)
 * LatLon_Valid
     Quality indicator flag providing a True/False value when the latitude
     and longitude for the provided location is the known location of the
     facility (if True), or if it is approximated based on imprecise or
     unavailable information (if False). If this flag is False, do not
     use the Latitude & Longitude values for geospatial analysis, such as
     determining whether the facility falls within certain political or
     accounting regions, as the actual facility may be a significant 
     distance from the point provided here.
 * CD119_2024
     FIPS code for the congressional district in which the facility 
     is located (if the facility location is known precisely - see 
     LatLon_Valid).
 * CD119_2024_Name
     The name of the congressional district in which the facility 
     is located (in format SS-00: [STATE ABBREV]-[DISTRICT NUMBER]).
     Not provided if LatLon_Valid is False.
 * US Senator 1: Name
     The name of the Senior US Senator for the state in which the
     facility is located. Not provided if LatLon_Valid is False.
 * US Senator 1: Party
     Party of the Senior US Senator for the state in which the
     facility is located. Not provided if LatLon_Valid is False.
 * US Senator 2: Name
     The name of the Junior US Senator for the state in which the
     facility is located. Not provided if LatLon_Valid is False.
 * US Senator 2: Party
     Party of the Junior US Senator for the state in which the
     facility is located. Not provided if LatLon_Valid is False.
 * US Representative Name
     Name of the US Congressional Representative for the district
     in which the facility is located. Not provided if 
     LatLon_Valid is False.
 * US Representative Party
     Party of the US Congressional Representative for the district
     in which the facility is located. Not provided if 
     LatLon_Valid is False.



socioeconomics.csv
------------------
Socioeconomic data used for calculating per-capita and fraction of GDP statistics. 
Real GDP is reported in 2023 Million USD.

Column descriptions:
 * State
     US State and D.C. 2-letter abbreviation
 * StateName
     US State and D.C. Name
 * quarter
     Reporting quarter in the format YYYY-QQ
 * population
     Total population estimate (individuals), interpolated to a
     quarterly frequency. Annual data is from the US Census Bureau, 
     then is interpolated to quarters between July 1 estimates. 
     For quarters more recent than the last available US Census
     estimate, the last available year-on-year growth rate is used
     to extrapolate the series.
 * real_gdp
    Quarterly real GDP (annualized)
 
federal_actual_investment_by_category.csv
-------------------------------
Total estimated quarterly federal investment via tax credits, grants, 
loans, and loan guarantees, reported in 2023 Million USD. Values are 
reported by the quarter in which activity eligible for tax credits occurs, 
or in which grants or loans are disbursed, not when they are announced or 
obligated.

See our methodology document, and especially the PDF download extended 
methodology at the bottom of the page) for more detail: 
https://www.cleaninvestmentmonitor.org/methodology
 
Column descriptions:
* Segment
     Market segment. One of "Manufacturing", "Energy and Industry", or
     "Retail." "Manufacturing" indicates an investment in facilities or
     capacity to produce GHG-reducing technology. "Energy and Industry"
     refers to the deployment of technologies that reduce GHG emissions
     in the bulk production of energy or industrial goods or that
     capture ambient carbon dioxide. "Retail" refers to the purchase and
     installation of technology by individual households and businesses.
* Category
     Federal expenditure category. One of "Clean Electricity Tax Credits"
     referring to those eligible under sections 45, 45U, 45Y, 48, and 48E; 
     "Emerging Climate Technology Tax Credits" including 45V, 45Q, 48C, 
     and 40B; "Advanced Manufacturing Tax Credits" including 45X, 48C and 
     48D; "Non-residential Distributed Energy Tax Credits" under section 48; 
     "Residential Energy & Efficiency Tax Credits" including 25C and 45L; 
     "Zero Emission Vehicle Tax Credits" including 30D and 45W; or 
     "Grants, Loans, and Loan Guarantees" which estimates outlays from 
     federal programs in IIJA and IRA intended to support the manufacture 
     and deployment of in-scope GHG-reducing technologies.
 * quarter
     The quarter in which activity eligible for a tax credit takes place
     or the Federal Government disburses a grant or loan or guarantees 
     a loan.
 * Total Federal Investment 
     Estimated quarterly investment via tax credits, grants, loans, and 
     loan guarantees. In general, tax 
     estimates assume 100% of the eligible credits were taken in 
     that quarter, including bonus tax credits as relevant. Outlays 
     estimates rely on Congressional Budget Office estimates of annual 
     outlay rates. 
 
federal_actual_investment_by_state.csv
-------------------------------
Total estimated federal investment by state via tax credits, grants, 
loans, and loan guarantees, reported in 2023 Million USD. Values are reported by 
the quarter in which activity eligible for tax credits occurs, or in which grants or
loans are disbursed, not when they are announced or obligated.

See our methodology document, and especially the PDF download extended 
methodology at the bottom of the page) for more detail: 
https://www.cleaninvestmentmonitor.org/methodology
 
Column descriptions:
 * State
     US State and D.C. 2-letter abbreviation
 * quarter
     The quarter in which activity eligible for a tax credit takes place
     or the Federal Government disburses a grant or loan or guarantees 
     a loan.
 * Total Federal Investment 
     Estimated quarterly investment via tax credits, grants, loans, and 
     loan guarantees. In general, tax estimates assume 100% of the eligible 
     credits were taken in that quarter, including bonus tax credits as relevant. 
     Outlays estimates rely on Congressional Budget Office estimates of annual 
     outlay rates. 
 * Federal Investment (per capita)
     Estimated federal investment, via tax credits, grants, loans, and 
     loan guarantees, per capita using total state population estimate.  
 * Federal investment as a fraction of state GDP
     Estimated federal investment, via tax credits, grants, loans, and 
     loan guarantees, reported as a fraction of state gross domestic 
     product.
 * population
     Total population estimate (individuals), interpolated to a
     quarterly frequency. Annual data is from the US Census Bureau,
     then is interpolated to quarters between July 1 estimates.
     For quarters more recent than the last available US Census
     estimate, the last available year-on-year growth rate is used
     to extrapolate the series.
 * State GDP (Annualized)
     Quarterly state real GDP, converted to an annualized basis
     (so the annual GDP is the average of four quarters, not the sum).


congressional_district_actual_investment_manufacturing_energy_and_industry.csv
---------------------------------------------------------------
Actual investment in Manufacturing and Energy & Industry segments 
by congressional district

Includes actual investment in manufacturing, utility-scale energy, and 
industrial facilities for which the precise location is known. Because 
facilities with imprecise location data and the Retail segment are excluded, 
see quarterly_actual_investment.csv for the most comprehensive list of actual 
investments. Investments are mapped to districts in effect for the 118th Congress.

Data are reported by segment, congressional district, and quarter
in 2023 million USD.

Column descriptions:
 * Segment
     Market segment. One of "Manufacturing" or "Energy and Industry.” 
     “Manufacturing" indicates an investment in facilities or capacity 
     to produce GHG-reducing technology. "Energy and Industry" refers 
     to the deployment of technologies that reduce GHG emissions in the 
     bulk production of energy or industrial goods or that capture ambient 
     carbon dioxide. 
 * State
     US State in which the investment was made
 * CD119_2024
     FIPS code for the congressional district.
 * CD119_2024_Name
     The name of the congressional district.
 * US Senator 1: Name
     The name of the Senior US Senator for the state.
 * US Senator 1: Party
     Party of the Senior US Senator for the state.
 * US Senator 2: Name
     The name of the Junior US Senator for the state.
 * US Senator 2: Party
     Party of the Junior US Senator for the state.
 * US Representative Name
     Name of the US Congressional Representative for the district.
 * US Representative Party
     Party of the US Congressional Representative for the district.
 * quarter
     The quarter in which the investment is estimated to take place in
     the format YYYY-QQ
 * Estimated_Actual_Quarterly_Expenditure
     Estimated quarterly investment, reported in 2023 Million USD.
     The time series of actual investment is based on reported investment totals, 
     extrapolated based on reported capacity and facility characteristics,
     or modeled based on known investments from similar facilities.
     Manufacturing, utility-scale energy, and industrial investments are 
     spread evenly over the duration of each facility's construction period.
     Construction start and end dates may be reported (e.g. in monthly 
     status filings, news sources or press releases) or may be estimated
     based on typical construction times for similar projects.

Technology Descriptions
=======================

This section lists all Technology fields covered in the
Clean Investment Monitor.

Manufacturing
-------------
* Batteries
* Critical Minerals
* Electrolyzers
* Solar
* Wind
* Fueling Equipment
* Zero Emission Vehicles

Energy and Industry
-------------------

* Carbon Management
* Cement
* Clean Fuels
* Hydrogen
* Iron & Steel 
* Pulp & Paper
* SAF
* Storage
* Solar
* Wind
* Nuclear
* Other

Retail
------
* Distributed Electricity and Storage
* Heat Pumps
* Zero Emission Vehicles

