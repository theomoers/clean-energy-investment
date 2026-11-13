"""Outcome-blind CIM product mapping under 26 CFR 1.45X-3 and -4 (TD 10010)."""
SOURCE = "https://www.irs.gov/irb/2024-51_IRB#TD-10010"


def classify(technology, product):
    """Classify product categories; tax basis, domestic production and sales are unobserved."""
    parts = {p.strip().lower() for p in product.split(",")}
    if technology in {"Electrolyzers", "Zero Emission Vehicles", "Fueling Equipment"}:
        return "outside", "Finished electrolyzer, vehicle or charger is outside the enumerated categories."
    if technology == "Batteries":
        return "clear", "Battery cells, modules and electrode active materials are enumerated components."
    if technology == "Wind" and parts <= {"blades", "towers", "nacelles"}:
        return "clear", "Listed wind component."
    if technology == "Solar" and parts <= {"cells", "modules", "wafers", "integrated cdte", "microinverters",
            "utility inverters", "residential inverters", "commercial inverters", "torque tubes"}:
        return "clear", "Photovoltaic component, listed inverter class or tracker torque tube."
    if technology == "Solar" and parts == {"backsheets"}:
        return "ambiguous", "Only qualifying polymeric backsheets meet the definition."
    if technology == "Critical Minerals" and parts <= {"lithium carbonate", "lithium hydroxide"}:
        return "clear", "Listed lithium conversion product; actual purity and taxpayer eligibility unknown."
    return "ambiguous", "Label lacks required processing, component detail, or describes a mixed product."
