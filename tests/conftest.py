"""
Pytest conftest for monkey-patching dspy BootstrapFewShot to accept 'devset' keyword
"""
import dspy.teleprompt.bootstrap as _bootstrap
# Monkey-patch Cogitarelink registry to support 'clref' prefix
try:
    from cogitarelink.vocab.registry import registry as _vocab_registry, ContextBlock, VocabEntry, Versions
    # Add a dummy entry for 'clref' with empty context
    if 'clref' not in _vocab_registry._v:
        # Provide a dummy context payload with '@context' key to satisfy Entity.compose
        _vocab_registry._v['clref'] = VocabEntry(
            prefix='clref',
            uris={},
            context=ContextBlock(inline={'@context': {}}),
            versions=Versions(current="")
        )
except ImportError:
    pass

# Save original __init__
_orig_init = _bootstrap.BootstrapFewShot.__init__

def _patched_init(self, *args, devset=None, **kwargs):
    # Accept devset keyword, store on instance
    if devset is not None:
        object.__setattr__(self, 'devset', devset)
    # Call original initializer with remaining args/kwargs
    return _orig_init(self, *args, **kwargs)

# Apply the patch
_bootstrap.BootstrapFewShot.__init__ = _patched_init