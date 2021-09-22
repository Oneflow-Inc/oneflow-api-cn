import docreset as m


class CFoo:
    r"""just a example"""

    def foo(self):
        pass


assert CFoo.__doc__ == r"""just a example"""

id_print = id(print)

r = m.reset_docstr(print, "hello")
assert id_print == id(r)

assert r.__doc__ == "hello"
