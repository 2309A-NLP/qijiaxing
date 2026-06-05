"""检查 SentenceTransformer Transformer 类的 auto_model 属性"""
from sentence_transformers.models import Transformer
import inspect

src = inspect.getsource(Transformer.__init__)
# Find auto_model related code
for i, line in enumerate(src.split('\n')):
    if 'auto_model' in line.lower():
        print(f'  {line.strip()}')

# Check if auto_model is a property
print(f'\nauto_model is property: {isinstance(type(Transformer).__dict__.get(\"auto_model\", None), property)}')

# Check the class dict for auto_model related attributes
for key in dir(Transformer):
    if 'auto_model' in key.lower():
        obj = getattr(Transformer, key, None) if key in type(Transformer).__dict__ else None
        print(f'{key}: {type(obj).__name__}')
