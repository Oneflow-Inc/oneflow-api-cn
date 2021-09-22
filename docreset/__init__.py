import os
from doctest import DocTestParser, DebugRunner, DocTestRunner
from ._docreset import _reset_docstr


FLAG_TEST = os.getenv("DOCTEST")
FLAG_BUILD = os.getenv("DOCBUILD")


def _test_docstr(docstr, verbose=True, optionflags=0, raise_on_error=True):
    parser = DocTestParser()
    if raise_on_error:
        runner = DebugRunner(verbose=verbose, optionflags=optionflags)
    else:
        runner = DocTestRunner(verbose=verbose, optionflags=optionflags)
    test = parser.get_doctest(docstr, {}, __name__, __file__, 0)
    runner.run(test)


def reset_docstr(o, docstr):
    global FLAG_TEST
    global FLAG_BUILD

    if FLAG_TEST:
        _test_docstr(docstr)

    if FLAG_BUILD:
        if type(o) == type:
            assert hasattr(o, "__doc__"), str(o) + " does not have a docstring!"
            setattr(o, "__doc__", docstr)
            return o
        else:
            return _reset_docstr(o, docstr)
