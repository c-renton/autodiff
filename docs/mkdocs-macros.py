def define_env(env):
    """
    This is the hook for the functions

    - env.variables: the dictionary that contains the variables
    - env.macro: a decorator function, to declare a macro.

    Note: mkdocs-macros-plugin >= 1.0 renamed the module hook from
    declare_variables(variables, macro) to define_env(env), with the two
    former positional arguments now available as env.variables and
    env.macro respectively.
    """

    @env.macro
    def inputcode(filename, language, startline=0, endline=None):
        filename = '../' + filename  # file path must be given relative to root directory
        f = open(filename, 'r')
        if startline != 0 or endline != None:
            lines = f.readlines()
            lines = lines[startline:endline]
            text = "".join(lines)
        else:
            text = f.read()
        textblock = f'```{language}\n{text}\n```'
        return textblock

    @env.macro
    def inputcpp(filename, startline=0, endline=None):
        return inputcode(filename, 'cpp', startline, endline)
