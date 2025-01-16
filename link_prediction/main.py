# Predict the missing links of a network using a KGE model proxy (due to time reasons artificial data is used)
#
# Voting methods from social choice theory, i.e. Ensemble learning
#
# Prints a latex table with the results and

import numpy as np

from typing import Iterable


from kge import KGE_dummy
from kge_models import *
from query import Query
from voting_methods import Majority, Borda, Range
from plot_graphs import plot_graph

# Presentation specific stuff
from presentation import plot_graph_presentation


def latex_table(query: Query,
                entity_of_interest: str,
                entities: Iterable[str],
                model_preds: np.ndarray,
                kge_models,
                voting_methods,
                k: int = 4) -> str:
    """ Converts the predicted values to a latex table

    Input:
        model_preds.shape = (n_models, n_entities)
        unsorted float values for query, e.g. (?, Orbits, Sun)
    """
    n_cols = len(kge_models)+len(voting_methods)+1

    # Calculations
    ranking_idz = np.argsort(-model_preds, axis=-1)
    voting_val = [voting_methods[i](model_preds) for i in range(len(voting_methods))]
    # Quickfix since voting methods don't yet work

    # voting_idz = [np.argsort(-voting_val[i], axis=-1) for i in range(len(voting_methods))]
    # print(voting_methods)
    # print(f"ranking_idz {ranking_idz}")

    # Shorten the name of the entities
    shrink_names_dict = {"Curiosity Rover": "Rover",
                  "Jupiter": "Jup."}
    for i, e in enumerate(entities):
        if e in shrink_names_dict:
            entities[i] = shrink_names_dict[e]

    # Latex
    table = "\\begin{table} \n"
    table += "\\centering \n"
    table += "\\begin{tabular}{" + "r|"*(n_cols-1) + "r}\n"
    
    # Super-Header
    table += f"&\\multicolumn{{{len(kge_models)}}}" + "{|c|}{Models}&\n"
    table += f"\\multicolumn{{{len(voting_methods)}}}" + "{|c}{Voting Methods using Models $h_i$}" + r"\\"
    table += "\n \\hline \n"
    # Header
    table += f"Rank &"
    for i_col in range(n_cols-1):
        if i_col < len(kge_models):
            table += f"$h_{i_col}(q, e)$ &"
        else:
            table += f"{voting_methods[i_col-len(kge_models)]} &"
    table = table[:-2] + r"\\" + "\n\\hline\n"

    # Data
    for i_rank in range(len(entities)):
        table += f"{i_rank+1} &"
        for i_model in range(len(kge_models)):
            idx = ranking_idz[i_model, i_rank]
            e = entities[idx]
            val = model_preds[i_model, idx]
            table += f"{e} ({val:.1f})&\t\t"
        
        for i_vot_mod in range(len(voting_methods)):
            # e = entities[voting_idz[i_rank]]
            # val = voting_val[i_vot_mod, voting_idz[i_rank]]
            table += f"{e} ({val:.1f}) &"
        
        table = table[:-2] + r"\\" + "\n"
        if i_rank == k-1:
            table += r"\hline \hline" + "\n"
    
    # Latex
    table += r"\end{tabular}" + "\n"
    table += r"\caption{Sorted values of $h_i(tr(q,e))$ with $q=\langle ?, \text{orbits}, \text{sun} \rangle$ where we want to predict " + entity_of_interest + r"}" + "\n"
    table += r"\end{table}"

    return table


def main(entities, train_relations):
    # Prediction what orbits the sun
    test_queries = [Query("Sun", "orbits", head_is_missing=True)]
    entities_of_interest = ["Moon"]
    truth_probs = [0.4]

    # Define Models and Voting methods
    kge_models = [KGE_model_1(entities),
                  KGE_model_2(entities),
                  KGE_model_3(entities)]
        
    voting_methods = [Majority(), Borda(), Range()]
    
    # "Training" (just memorizing)
    for model in kge_models:
        model.fit(train_relations, [0.] * len(train_relations))

    # 
    model_preds = np.zeros((len(kge_models), len(test_queries), len(entities)))
    # Prediction
    for i, model in enumerate(kge_models):
        model_preds[i] = model.predict_w_truth_prob(test_queries, truth_probs, entities_of_interest)
        idz = np.argsort(-model_preds[i])
    
    # Since only one query for table, remove inner dim
    # (n_models, n_queries, n_entities) -> (n_models, n_entities)
    model_preds = model_preds.squeeze()
    latex_str = latex_table(query=test_queries[0],
                            entity_of_interest=entities_of_interest[0],
                            entities=entities,
                            model_preds=model_preds,
                            kge_models=kge_models,
                            voting_methods=voting_methods)
    # print(latex_str)
    # save to file
    with open("table.tex", "w") as f:
        f.write(latex_str)
    
    

if __name__ == "__main__":
    # Draw nodes (entities) # 5:3.5
    entities_dict = {
        "Earth": (3, -0.5),
        # "Saturn": (-0.5, 1.5),
        "Jupiter": (-2, 1.5),
        "Mars": (0, 2),
        "Curiosity Rover": (3.5, 1.5),
        "Hubble": (-1.5, -1.5),
        # "James Webb": (-2, -1.5),
        # "Sirius": (-3, 0),
        "Moon": (2, -1.5),
        "Sun": (0, 0),
        # "Titan": (0.5, 2.5)
    }
    entities = list(entities_dict.keys())

    # Triplets (start, end, relation, truth_prob)
    train_relations = [
        ("Moon", "orbits", "Earth"),
        ("Earth", "orbits", "Sun"),
        ("Mars", "orbits", "Sun"),
        ("Jupiter", "orbits", "Sun"),
        # ("Saturn", "orbits", "Sun"),
        ("Hubble", "orbits", "Earth"),
        ("Hubble", "observes", "Jupiter"),
        # ("James Webb", "observes", "Saturn"),
        ("Hubble", "observes", "Mars"),
        # ("James Webb", "observes", "Sirius"),
        # ("James Webb", "observes", "Jupiter"),
        # ("James Webb", "observes", "Saturn"),
        ("Earth", "observes", "Mars"),
        ("Curiosity Rover", "observes", "Mars"),
        # ("Titan", "orbits", "Saturn"),

    ]
    test_relations = [
        # ("Earth", "observes", "Curiosity Rover", 0.9),
        # ("Curiosity Rover", "orbits", "Sun", 0.4),
        # ("Sun", "orbits", "Hubble", 0.1),
        ("Moon", "orbits", "Sun", 0.7),
        ("Jupiter", "orbits", "Sun", 0.7),
        ("Mars", "orbits", "Sun", 0.7),
        ("Earth", "orbits", "Sun", 0.7),
        ("Curiosity Rover", "orbits", "Sun", 0.7),
        ("Moon", "orbits", "Sun", 0.7),
        ("Hubble", "orbits", "Sun", 0.7),
        # ("Hubble", "observes", "Sirius", 0.9),
    ]

    plot_graph(entities_dict, train_relations, test_relations)
    main(entities, train_relations)
    
    plot_graph_presentation(entities_dict, train_relations, test_relations=[],
                            fname="wout_test_queries")

    plot_graph_presentation(entities_dict, train_relations=[], test_relations=test_relations,
                            fname="example query",
                            show_pm_glyphs=False)





        