
#Returns a list of entities based on the label given.
def labels(option):

    # create a dictionary with your entity labels.  By defining which label goes with each entity, we can build a list based on label.
    label_dictionary = {
        'Total': ['orderform',],
        'Subtotal': ['orderform',],
        'Shipping': ['orderform',],
        'Vendor': ['orderform','pi',],
        'Institution': ['pi','resume',],
        'Education': ['pi','resume',],
        'Company': ['pi','resume',],
        'Phone Number': ['pi','resume',],
        'Address': ['pi','resume',],
        'Department': ['pi',],
        "Medical Institution": ['pi','resume',],
        "Institution Name": ['pi','resume',],
        "Doctor Name": ['pi',],
        "Supervisor Name": ['pi',],
        "person": ['pi','resume',],      # people, including fictional characters
        "fac": ['pi','resume',],         # buildings, airports, highways, bridges
        "org": ['pi','resume',],         # organizations, companies, agencies, institutions
        "gpe": ['pi','resume',],         # geopolitical entities like countries, cities, states
        "loc": ['pi','resume',],         # non-gpe locations
        "product": ['pi','resume','orderform',],     # vehicles, foods, appareal, appliances, software, toys 
        "event": ['pi','resume',],       # named sports, scientific milestones, historical events
        "work_of_art": ['pi','resume','orderform',], # titles of books, songs, movies
        "law": ['pi','resume',],         # named laws, acts, or legislations
        "language": ['pi','resume',],    # any named language
        "date": ['pi','resume','orderform',],        # absolute or relative dates or periods
        "time": ['pi','resume','orderform',],        # time units smaller than a day
        "percent": ['orderform',],     # percentage (e.g., "twenty percent", "18%")
        "money": ['orderform',],       # monetary values, including unit
        "quantity": ['orderform',],    # measurements, e.g., weight or distance
    }

    return [label for label, op in label_dictionary.items() if option in op]

#Creates the assistant message for the api call.  The assistant message gives an example of how the LLM should respond.
def assisstant_message():
    return f"""
EXAMPLE:
    Text: 'In Germany, in 1440, goldsmith Johannes Gutenberg invented the movable-type printing press. His work led to an information revolution and the unprecedented mass-spread / 
    of literature throughout Europe. Modelled on the design of the existing screw presses, a single Renaissance movable-type printing press could produce up to 3,600 pages per workday.'
    {{
        "gpe": ["Germany", "Europe"],
        "date": ["1440"],
        "person": ["Johannes Gutenberg"],
        "product": ["movable-type printing press"],
        "event": ["Renaissance"],
        "quantity": ["3,600 pages"],
        "time": ["workday"]
    }}
--"""
def system_message(labels_list):
    #print(type(labels_list))
    types=", ".join(labels_list)
    return f"""
You are an expert in Natural Language Processing. Your task is to identify common Named Entities (NER) in a given text.
The possible common Named Entities (NER) types are exclusively: ({types})."""
