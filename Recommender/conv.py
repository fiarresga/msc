import aiml
import datetime
from neo4j import GraphDatabase, RoutingControl
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

URI = "neo4j://localhost:7687"
AUTH=('neo4j', "M@userMaria")

cpa = pd.read_csv('cpa.csv')
met_values = pd.read_csv('met_values.csv')

kernel = aiml.Kernel()
kernel.learn("std-startup.xml")
kernel.respond("LOAD AIML B") 



def get_time():
    return datetime.datetime.now().strftime("%H:%M:%S")


def get_met_values(id):
    global id_global 
    id_global = id

    row = met_values[met_values['eid'] == int(id)]
    if not row.empty:
        met_min = int(row['met_min'].values[0])
        met_max = int(row['met_max'].values[0])
        return [str(num) for num in range(met_min, met_max + 1)]
    else:
        raise Exception


def get_explanation():
    row = met_values[met_values['eid'] == int(id_global)]
    if row['Predictions'].values[0] == 0:
        return 'You are currently considered to be Active.'
    elif row['Predictions'].values[0] == 1:
        return 'You are currently considered to be Sedentary.'



def get_recommendations(type_name, keywords, id):
    met_values = get_met_values(id)
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        records, _, _ = driver.execute_query(
            '''
            MATCH (t:Type {name: $type_name})<-[:HAS_TYPE]-(s:SAC)
                -[:HAS_MET]->(m:MET),
                (s)-[:HAS_KEYWORD]->(k:Keyword)
            WHERE k.name IN $keywords
            AND m.name IN $met_values
            RETURN t, s, m, k
            ''',
            parameters_={
                "type_name": type_name,
                "keywords": keywords,
                "met_values": met_values
            },
            database_="neo4j",
            routing_=RoutingControl.READ
        )
        return records
    

def handle_recommendations():
    global last_records
    id_input = input("Enter Participant ID: ")
    type_input = input("Enter the type: ")
    keywords_input = input("Enter keywords (comma-separated): ")
    
    keywords = [kw.strip() for kw in keywords_input.split(",")]
    
    records = get_recommendations(type_input, keywords, id_input)
    last_records = records

    if not records:
        return "No recommendations found."
    
    cpa = pd.read_csv('cpa.csv')

    for record in records:
        code = record['s']['code']
        match = cpa[cpa['SAC'] == int(code)]
        print(match.iloc[0]['SAC'], ", MET:", match.iloc[0]['MET'], ", Activity: ", match.iloc[0][' Description'])



def show_plot(records):
    G = nx.DiGraph()

    for record in records:
        t = record["t"]
        s = record["s"]
        m = record["m"]
        k = record["k"]

        G.add_node(t.element_id, label=t["name"], type="Type")
        G.add_node(s.element_id, label=s["code"], type="SAC")
        G.add_node(m.element_id, label=m["name"], type="MET")
        G.add_node(k.element_id, label=k["name"], type="Keyword")

        G.add_edge(s.element_id, t.element_id, label="HAS_TYPE")
        G.add_edge(s.element_id, m.element_id, label="HAS_MET")
        G.add_edge(s.element_id, k.element_id, label="HAS_KEYWORD")


    pos = nx.spring_layout(G, seed=42)  

    labels = nx.get_node_attributes(G, 'label')
    edge_labels = nx.get_edge_attributes(G, 'label')

    plt.figure(figsize=(10, 7))
    nx.draw(G, pos, labels=labels, with_labels=True, node_size=2000, node_color='lightblue', font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.title("Neo4j Network Visualization")
    plt.show()


def handle_show_plot():
    try:
        if last_records:
            pass
    except:
        return "No data to plot. Please run a recommendation first."
    show_plot(last_records)
    return "Plot displayed."


command_handlers = {
    "GET_TIME": get_time,
    "GET_RECOMMENDATIONS": handle_recommendations,
    "SHOW_PLOT": handle_show_plot,
    "WHY": get_explanation,
}



while True:
    user_input = input("Enter your message >> ") 
    response = kernel.respond(user_input)
    
    if response in command_handlers:
        records = command_handlers[response]()
        print(records, '\n')
    else:
        print(response)
