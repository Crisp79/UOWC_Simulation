from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wlexpr, wl

kernel_path = 'path to kernel'

with WolframLanguageSession(kernel_path) as session:
    # Use wlexpr to write the math exactly like you would in Mathematica
    result = session.evaluate(wlexpr('Expand[(x + y)^2]'))
    print(f"Result: {result}")