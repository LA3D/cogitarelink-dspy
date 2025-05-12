# dspy.Signature

##  `` `dspy.Signature`

Bases: `BaseModel`

### Functions

####  `` `append(name, field, type_=None) -> Type[Signature]` `classmethod`

Source code in `dspy/signatures/signature.py`


|

```
@classmethod
def append(cls, name, field, type_=None) -> Type["Signature"]:
    return cls.insert(-1, name, field, type_)

```  
  
---|---  
  
####  `` `delete(name) -> Type[Signature]` `classmethod`

Source code in `dspy/signatures/signature.py`

|

```
@classmethod
def delete(cls, name) -> Type["Signature"]:
    fields = dict(cls.fields)

    if name in fields:
        del fields[name]
    else:
        raise ValueError(f"Field `{name}` not found in `{cls.__name__}`.")

    return Signature(fields, cls.instructions)

```  
  
---|---  
  
####  `` `dump_state()` `classmethod`

Source code in `dspy/signatures/signature.py`


```
@classmethod
def dump_state(cls):
    state = {"instructions": cls.instructions, "fields": []}
    for field in cls.fields:
        state["fields"].append(
            {
                "prefix": cls.fields[field].json_schema_extra["prefix"],
                "description": cls.fields[field].json_schema_extra["desc"],
            }
        )

    return state

```  
  
---|---  
  
####  `` `equals(other) -> bool` `classmethod`

Compare the JSON schema of two Signature classes.

Source code in `dspy/signatures/signature.py`

|

```
@classmethod
def equals(cls, other) -> bool:
    """Compare the JSON schema of two Signature classes."""
    if not isinstance(other, type) or not issubclass(other, BaseModel):
        return False
    if cls.instructions != other.instructions:
        return False
    for name in cls.fields.keys() | other.fields.keys():
        if name not in other.fields or name not in cls.fields:
            return False
        if cls.fields[name].json_schema_extra != other.fields[name].json_schema_extra:
            return False
    return True

```  
  
---|---  
  
####  `` `insert(index: int, name: str, field, type_: Type = None) -> Type[Signature]` `classmethod`

Source code in `dspy/signatures/signature.py`


|

```
@classmethod
def insert(cls, index: int, name: str, field, type_: Type = None) -> Type["Signature"]:
    # It's possible to set the type as annotation=type in pydantic.Field(...)
    # But this may be annoying for users, so we allow them to pass the type
    if type_ is None:
        type_ = field.annotation
    if type_ is None:
        type_ = str

    input_fields = list(cls.input_fields.items())
    output_fields = list(cls.output_fields.items())

    # Choose the list to insert into based on the field type
    lst = input_fields if field.json_schema_extra["__dspy_field_type"] == "input" else output_fields
    # We support negative insert indices
    if index < 0:
        index += len(lst) + 1
    if index < 0 or index > len(lst):
        raise ValueError(
            f"Invalid index to insert: {index}, index must be in the range of [{len(lst) - 1}, {len(lst)}] for "
            f"{field.json_schema_extra['__dspy_field_type']} fields, but received: {index}.",
        )
    lst.insert(index, (name, (type_, field)))

    new_fields = dict(input_fields + output_fields)
    return Signature(new_fields, cls.instructions)

```  
  
---|---  
  
####  `` `load_state(state)` `classmethod`

Source code in `dspy/signatures/signature.py`

`

|

```
@classmethod
def load_state(cls, state):
    signature_copy = Signature(deepcopy(cls.fields), cls.instructions)

    signature_copy.instructions = state["instructions"]
    for field, saved_field in zip(signature_copy.fields.values(), state["fields"]):
        field.json_schema_extra["prefix"] = saved_field["prefix"]
        field.json_schema_extra["desc"] = saved_field["description"]

    return signature_copy

```  
  
---|---  
  
####  `` `prepend(name, field, type_=None) -> Type[Signature]` `classmethod`

Source code in `dspy/signatures/signature.py`

|

```
@classmethod
def prepend(cls, name, field, type_=None) -> Type["Signature"]:
    return cls.insert(0, name, field, type_)

```  
  
---|---  
  
####  `` `with_instructions(instructions: str) -> Type[Signature]` `classmethod`

Source code in `dspy/signatures/signature.py`


|

```
@classmethod
def with_instructions(cls, instructions: str) -> Type["Signature"]:
    return Signature(cls.fields, instructions)

```  
  
---|---  
  
####  `` `with_updated_fields(name, type_=None, **kwargs) -> Type[Signature]` `classmethod`

Create a new Signature class with the updated field information.

Returns a new Signature class with the field, name, updated with fields[name].json_schema_extra[key] = value.

Parameters:

Name | Type | Description | Default  
---|---|---|---  
`name` |  |  The name of the field to update. |  _required_  
`type_` |  |  The new type of the field. |  `None`  
`**kwargs` |  |  The new values for the field. |  `{}`  
  
Returns:

Type | Description  
---|---  
`Type[Signature]` |  A new Signature class (not an instance) with the updated field information.  
Source code in `dspy/signatures/signature.py`


|

```
@classmethod
def with_updated_fields(cls, name, type_=None, **kwargs) -> Type["Signature"]:
    """Create a new Signature class with the updated field information.

    Returns a new Signature class with the field, name, updated
    with fields[name].json_schema_extra[key] = value.

    Args:
        name: The name of the field to update.
        type_: The new type of the field.
        **kwargs: The new values for the field.

    Returns:
        A new Signature class (not an instance) with the updated field information.
    """
    fields_copy = deepcopy(cls.fields)
    # Update `fields_copy[name].json_schema_extra` with the new kwargs, on conflicts
    # we use the new value in kwargs.
    fields_copy[name].json_schema_extra = {
        **fields_copy[name].json_schema_extra,
        **kwargs,
    }
    if type_ is not None:
        fields_copy[name].annotation = type_
    return Signature(fields_copy, cls.instructions)

```  