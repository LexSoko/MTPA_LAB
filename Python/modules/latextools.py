import pandas as pd
import uncertainties as un
from uncertainties.core import ufloat

def latextable(data:pd.DataFrame,format):
    print("                      ")
    print("Latextable generation:")
    print("----------------------")
    head = data.columns
    print("Data:")
    print("......................")
    print(data)
    f = format * len(data.T)
    table_string = ""
    
    #begin table
    table_string += "\\" + "begin{table}[H] \n"
    table_string += "\label{} \n\centering \n\captionof{table}{} \n"
    table_string += "\\"+ "begin{tabular}{"+ f +"} \hline \n"

    # create table header
    #for variable in head:
    #    table_string += variable + "&"
    #table_string = table_string[0:-1]       # delete last "&"
    #table_string += " \\\ \hline \n"
    
    # create table data fields
    for index, row in data.iterrows():          # row is now a pandas series of a row
        for element in row:                     # element is now a single table entry
            table_string += "$"+  ('{:L}'.format(element) if isinstance(element,un.UFloat) else str(element))  +"$" + "&"
        table_string = table_string[0:-1]       # delete last "&"
        table_string += "\\\ \hline \n"
        
    #end table
    table_string +="\end{tabular} \n\end{table}"
    
    print("Table:")
    print("......................")
    print(table_string)
    return table_string

def latextable_custom(width: int , height: int, format: str):
    print("                      ")
    print("Latextable generation:")
    print("----------------------")
    head = width
    
    f = format * width
    table_string = ""
    
    #begin table
    table_string += "\\" + "begin{table}[H] \n"
    table_string += "\label{} \n\centering \n\caption{} \n"
    table_string += "\\"+ "begin{tabular}{"+ f +"} \hline \n"

    
    
    # create table data fields
    for row in range(height):          # row is now a pandas series of a row
        for element in range(width):                     # element is now a single table entry
            table_string += " " + "&"
        table_string = table_string[0:-1]       # delete last "&"
        table_string += "\\\ \hline \n"
        
    #end table
    table_string +="\end{tabular} \n\end{table}"
    
    print("Table:")
    print("......................")
    print(table_string)
    return table_string

#latextable_custom(2,9,"c")