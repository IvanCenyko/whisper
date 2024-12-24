import whisper
import google.generativeai as genai
import os, sys
import dotenv

# key de Gemini en .env
dotenv.load_dotenv(".env")
gemini_key = os.getenv("GEMINI_KEY")

# mando la salida estandar a null para que no imprima warnings molestos
real_stdout = sys.stdout
real_stderr = sys.stderr

sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

# configuro gemini 2.0
genai.configure(api_key=gemini_key)
model_gen = genai.GenerativeModel("gemini-2.0-flash-exp")

# cargo whisper en small con CPU
model = whisper.load_model("small", device="cpu")
# le pido que transcriba el audio
result = model.transcribe("audio_maxi_3.ogg")

# redirijo la salida estandar de nuevo a consola
sys.stdout = real_stdout
sys.stderr = real_stderr

# le pido a Gemini que corrija errores de lo generado con Whisper
response = model_gen.generate_content(f"""
1. El siguiente texto fue generado por Whisper AI, una IA de transcripcion de audio a texto, podria tener errores 
de ortografia o gramaticales. Te pido que los arregles, no cambies terminos ni jergas especificas, concentrate en corregir
SOLAMENTE gramatica, o palabras que parecen ser dos distintas pero estan unidas. Responde directamente la correccion
sin agregar nada mas. Ante la duda, deja algo como esta y no lo cambies. Debajo de los guiones estara el mensaje. Como ayuda,
puedo decirte que los hablantes son argentinos, por tanto puede haber jerga (por ejemplo decir volue en vez de boludo)
propia que corregir porque Whisper no la entendio. Gracias.
2. Si tiene mas de 100 palabras, haz un mini resumen debajo aclarando que lo es. Para el resumen, puedes usar citas
del texto, si no estas seguro de la tematica del mensaje, aclaralo, pero es importante que intentes interpretarlo de alguna forma.
------------------------------------------------------------------------------------------------
{result["text"]}""")

# printeo resultado
print(response.text)


