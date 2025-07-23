import pandas as pd
import tempfile
import csv
from groq import Groq

def preprocess_and_save(file_path):
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='utf-8', na_values=['NA', 'N/A', 'missing'])
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, na_values=['NA', 'N/A', 'missing'])
        else:
            return None, None, None, "Unsupported file format."

        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)

        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except:
                    pass

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode='w', newline='', encoding='utf-8') as temp_file:
            df.to_csv(temp_file.name, index=False, quoting=csv.QUOTE_ALL)
            return df, df.columns.tolist(), df.to_html(classes='table-auto w-full'), None
    except Exception as e:
        return None, None, None, str(e)

# ======= USER INPUT SECTION =======
file_path = "diabetes.csv"  
query = "How many patients have diabetes (Outcome = 1)?" 
groq_api_key = "" 

# ======= EXECUTION FLOW =======
if not groq_api_key:
    print("❌ Please provide your Groq API key.")
else:
    try:
        df, cols, df_html, err = preprocess_and_save(file_path)

        if err:
            print(f"❌ Error: {err}")
        else:
            print("✅ File successfully processed.")
            print("\n📊 Preview of first 5 rows:")
            print(df.head())

            if query:
                prompt = f"""
You are a Python data analyst. Given a pandas DataFrame named `df`, write Python code using pandas to answer this question:

Question: {query}

Only return the Python code (no explanation). Use 'result' as the final output variable.
"""

                client = Groq(api_key=groq_api_key)
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile"
                )

                code_generated = chat_completion.choices[0].message.content.strip("`python").strip("`")
                print("\n🤖 Generated Code:\n")
                print(code_generated)

                local_vars = {"df": df}
                exec(code_generated, {}, local_vars)

                result = local_vars.get("result", "⚠️ No result generated.")
                print("\n📈 Final Result:\n")
                if isinstance(result, pd.DataFrame):
                    print(result.to_string(index=False))
                else:
                    print(result)

    except Exception as e:
        print(f"❌ Exception occurred: {e}")
