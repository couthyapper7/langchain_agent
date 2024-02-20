import pandas as pd
from langchain.tools import BaseTool
import os
from django.conf import settings
base_dir = settings.BASE_DIR
csv_path = os.path.join(base_dir, 'data', 'autos.csv')
from langchain.tools import Tool
from abc import ABC, abstractmethod

class CSVSearchTool(Tool, ABC):  # Inherit from Tool and ABC (Abstract Base Class)
    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path

    @abstractmethod
    def _run(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        # Implement any common functionality here
        result = self._run(*args, **kwargs)
        # Implement any other common functionality here
        return result

class ConcreteCSVSearchTool(CSVSearchTool):
    def _run(self, *args, **kwargs):
        # Implement the actual functionality of the tool
        pass






class CSVSearchTool(BaseTool):
    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.columns = self.df.columns

    def format_response(self, df):
        if df.empty:
            return "Lo siento, no pude encontrar lo que estás buscando. ¿Hay algo más en lo que pueda ayudarte?"

        response_lines = []
        for _, row in df.iterrows():
            response_line = f"Tenemos un {row['MAKE']} modelo {row['MODEL']} del año {row['YEAR']}, con detalles {row['SUBMODEL']}."
            response_lines.append(response_line)

        return " ".join(response_lines)

    def run(self, prompt):
        prompt_lower = prompt.lower()

        matches = {}
        for column in self.columns:
            if column.lower() in prompt_lower:
                unique_values = self.df[column].unique()
                for value in unique_values:
                    if str(value).lower() in prompt_lower:
                        matches[column] = str(value)
                        break

        if matches:
            filtered_df = self.df
            for col, val in matches.items():
                filtered_df = filtered_df[filtered_df[col].str.lower() == val.lower()]

            return self.format_response(filtered_df)
        else:
            return "Lo siento, no tengo información suficiente para responder esa pregunta."
        

