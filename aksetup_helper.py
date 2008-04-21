import ez_setup

ez_setup.use_setuptools()

from setuptools import Extension

def setup(*args, **kwargs):
    from setuptools import setup
    import traceback
    try:
        setup(*args, **kwargs)
    except:
        traceback.print_exc()
        print "--------------------------------------------------------------------------"
        print "Sorry, your build failed. Try rerunning configure with different options."
        print "--------------------------------------------------------------------------"




# tools -----------------------------------------------------------------------
def flatten(list):
    """For an iterable of sub-iterables, generate each member of each 
    sub-iterable in turn, i.e. a flattened version of that super-iterable.

    Example: Turn [[a,b,c],[d,e,f]] into [a,b,c,d,e,f].
    """
    for sublist in list:
        for j in sublist:
            yield j




def humanize(sym_str):
    words = sym_str.lower().replace("_", " ").split(" ")
    return " ".join([word.capitalize() for word in words])




# siteconf handling -----------------------------------------------------------
def get_config():
    from setup import get_config_schema
    schema = get_config_schema()

    if not schema.have_config() and not schema.have_global_config():
        print "********************************************************"
        print "*** I have detected that you have not run configure."
        print "********************************************************"
        print "*** Additionally, no global config files were found."
        print "*** I will go ahead with the default configuration."
        print "*** In all likelihood, this will not work out."
        print "*** "
        print "*** See README_SETUP.txt for more information."
        print "*** "
        print "*** If the build does fail, just re-run configure with the"
        print "*** correct arguments, and then retry. Good luck!"
        print "********************************************************"
        print "*** HIT Ctrl-C NOW IF THIS IS NOT WHAT YOU WANT"
        print "********************************************************"

        delay = 10

        from time import sleep
        import sys
        while delay:
            sys.stdout.write("Continuing in %d seconds...   \r" % delay)
            sys.stdout.flush()
            delay -= 1
            sleep(1)

    return schema.read_config()




def hack_distutils():
    # hack distutils.sysconfig to eliminate debug flags
    # stolen from mpi4py
    import sys
    if not sys.platform.lower().startswith("win"):
        from distutils import sysconfig

        cvars = sysconfig.get_config_vars()
        cflags = cvars.get('OPT')
        if cflags:
            cflags = cflags.split()
            for bad_prefix in ('-g', '-O', '-Wstrict-prototypes'):
                for i, flag in enumerate(cflags):
                    if flag.startswith(bad_prefix):
                        cflags.pop(i)
                        break
                if flag in cflags:
                    cflags.remove(flag)
            cflags.append("-O3")
            #cflags.append("-g")
            cvars['OPT'] = str.join(' ', cflags)
            cvars["CFLAGS"] = cvars["BASECFLAGS"] + " " + cvars["OPT"]




# configure guts --------------------------------------------------------------
def default_or(a, b):
    if a is None:
        return b
    else:
        return a




class ConfigSchema:
    def __init__(self, options, conf_file="siteconf.py"):
        self.optdict = dict((opt.name, opt) for opt in options)
        self.options = options
        self.conf_file = conf_file

        from os.path import expanduser
        self.user_conf_file = expanduser("~/.aksetup-defaults.py")

        import sys
        if not sys.platform.lower().startswith("win"):
            self.global_conf_file = expanduser("/etc/aksetup-defaults.py")
        else:
            self.global_conf_file = None

    def get_default_config(self):
        return dict((opt.name, opt.default) 
                for opt in self.options)
        
    def read_config_from_pyfile(self, filename):
        result = {}
        filevars = {}
        execfile(filename, filevars)

        for key, value in filevars.iteritems():
            if key in self.optdict:
                result[key] = value

        return result

    def update_conf_file(self, filename, config):
        result = {}
        filevars = {}

        try:
            execfile(filename, filevars)
        except IOError:
            pass

        del filevars["__builtins__"]

        for key, value in config.iteritems():
            if value is not None:
                filevars[key] = value

        keys = filevars.keys()
        keys.sort()

        outf = open(filename, "w")
        for key in keys:
            outf.write("%s = %s\n" % (key, repr(filevars[key])))
        outf.close()

        return result

    def update_user_config(self, config):
        self.update_conf_file(self.user_conf_file, config)

    def update_global_config(self, config):
        if self.global_conf_file is not None:
            self.update_conf_file(self.global_conf_file, config)

    def get_default_config_with_files(self):
        result = self.get_default_config()

        import os
        
        confignames = []
        if self.global_conf_file is not None:
            confignames.append(self.global_conf_file)
        confignames.append(self.user_conf_file)

        for fn in confignames:
            if os.access(fn, os.R_OK):
                result.update(self.read_config_from_pyfile(fn))

        return result

    def have_global_config(self):
        import os
        return (os.access(self.user_conf_file, os.R_OK) or 
                os.access(self.global_conf_file, os.R_OK))

    def have_config(self):
        import os
        return os.access(self.conf_file, os.R_OK)

    def read_config(self, warn_if_none=True):
        import os
        result = self.get_default_config_with_files()
        if os.access(self.conf_file, os.R_OK):
            filevars = {}
            execfile(self.conf_file, filevars)

            for key, value in filevars.iteritems():
                if key in self.optdict:
                    result[key] = value
                elif key == "__builtins__":
                    pass
                else:
                    raise KeyError, "invalid config key in %s: %s" % (
                            self.conf_file, key)

        return result

    def add_to_configparser(self, parser, def_config=None):
        if def_config is None:
            def_config = self.get_default_config_with_files()

        for opt in self.options:
            default = default_or(def_config.get(opt.name), opt.default)
            opt.add_to_configparser(parser, default)

    def get_from_configparser(self, options):
        result = {}
        for opt in self.options:
            result[opt.name] = opt.take_from_configparser(options)
        return result

    def write_config(self, config):
        outf = open(self.conf_file, "w")
        for opt in self.options:
            value = config[opt.name]
            if value is not None:
                outf.write("%s = %s\n" % (opt.name, repr(config[opt.name])))
        outf.close()

    def make_substitutions(self, config):
        return dict((opt.name, opt.value_to_str(config[opt.name]))
                for opt in self.options)








class Option(object):
    def __init__(self, name, default=None, help=None):
        self.name = name
        self.default = default
        self.help = help

    def as_option(self):
        return self.name.lower().replace("_", "-")

    def metavar(self):
        last_underscore = self.name.rfind("_")
        return self.name[last_underscore+1:]

    def get_help(self, default):
        result = self.help
        if self.default:
            result += " (default: %s)" % self.value_to_str(
                    default_or(default, self.default))
        return result

    def value_to_str(self, default):
        return default 

    def add_to_configparser(self, parser, default=None):
        default = default_or(default, self.default)
        default_str = self.value_to_str(default)
        parser.add_option(
            "--" + self.as_option(), dest=self.name, 
            default=default,
            metavar=self.metavar(), help=self.get_help(default))

    def take_from_configparser(self, options):
        return getattr(options, self.name)

class Switch(Option):
    def add_to_configparser(self, parser, default=None):
        option = self.as_option()

        if not isinstance(self.default, bool):
            raise ValueError, "Switch options must have a default"

        if default is None:
            default = self.default

        if default:
            action = "store_false"
        else:
            action = "store_true"
            
        parser.add_option(
            "--" + self.as_option(), 
            dest=self.name, 
            help=self.get_help(default), 
            default=default,
            action=action)

class StringListOption(Option):
    def value_to_str(self, default):
        if default is None:
            return None

        return ",".join([str(el) for el in default])

    def get_help(self, default):
        return Option.get_help(self, default) + " (several ok)"

    def take_from_configparser(self, options):
        opt = getattr(options, self.name)
        if opt is None:
            return None
        else:
            if opt:
                return opt.split(",")
            else:
                return []


class IncludeDir(StringListOption):
    def __init__(self, lib_name, default=None, human_name=None, help=None):
        StringListOption.__init__(self, "%s_INC_DIR" % lib_name, default,
                help=help or ("Include directories for %s" 
                % (human_name or humanize(lib_name))))

class LibraryDir(StringListOption):
    def __init__(self, lib_name, default=None, human_name=None, help=None):
        StringListOption.__init__(self, "%s_LIB_DIR" % lib_name, default,
                help=help or ("Library directories for %s"
                % (human_name or humanize(lib_name))))

class Libraries(StringListOption):
    def __init__(self, lib_name, default=None, human_name=None, help=None):
        StringListOption.__init__(self, "%s_LIBNAME" % lib_name, default,
                help=help or ("Library names for %s (without lib or .so)" 
                % (human_name or humanize(lib_name))))







def configure_frontend():
    from optparse import OptionParser

    from setup import get_config_schema
    schema = get_config_schema()
    if schema.have_config():
        print "************************************************************"
        print "*** I have detected that you have already run configure."
        print "*** I'm taking the configured values as defaults for this"
        print "*** configure run. If you don't want this, delete the file"
        print "*** %s." % schema.conf_file
        print "************************************************************"

    import sys

    description = "generate a configuration file for this software package"
    parser = OptionParser(description=description)
    parser.add_option(
	    "--python-exe", dest="python_exe", default=sys.executable,
	    help="Which Python interpreter to use", metavar="PATH")

    parser.add_option("--prefix", default=None,
	    help="Ignored")
    parser.add_option("--enable-shared", help="Ignored", action="store_false")
    parser.add_option("--disable-static", help="Ignored", action="store_false")
    parser.add_option("--update-user", help="Update user config file (%s)" % schema.user_conf_file, 
            action="store_true")
    parser.add_option("--update-global", help="Update global config file (%s)" % schema.global_conf_file, 
            action="store_true")

    schema.add_to_configparser(parser, schema.read_config())

    options, args = parser.parse_args()

    config = schema.get_from_configparser(options)
    schema.write_config(config)

    if options.update_user:
        schema.update_user_config(config)

    if options.update_global:
        schema.update_global_config(config)

    import os
    if os.access("Makefile.in", os.F_OK):
        substs = schema.make_substitutions(config)
        substs["PYTHON_EXE"] = options.python_exe

        substitute(substs, "Makefile")




def substitute(substitutions, fname):
    import re
    var_re = re.compile(r"\$\{([A-Za-z_0-9]+)\}")
    string_var_re = re.compile(r"\$str\{([A-Za-z_0-9]+)\}")

    fname_in = fname+".in"
    lines = open(fname_in, "r").readlines()
    new_lines = []
    for l in lines:
        made_change = True
        while made_change:
            made_change = False
            match = var_re.search(l)
            if match:
                varname = match.group(1)
                l = l[:match.start()] + str(substitutions[varname]) + l[match.end():]
                made_change = True

            match = string_var_re.search(l)
            if match:
                varname = match.group(1)
                subst = substitutions[varname]
                if subst is None:
                    subst = ""
                else:
                    subst = '"%s"' % subst

                l = l[:match.start()] + subst  + l[match.end():]
                made_change = True
        new_lines.append(l)
    new_lines.insert(1, "# DO NOT EDIT THIS FILE -- it is generated by configure\n")
    import sys
    new_lines.insert(2, "# %s\n" % (" ".join(sys.argv)))
    open(fname, "w").write("".join(new_lines))

    from os import stat, chmod
    infile_stat_res = stat(fname_in)
    chmod(fname, infile_stat_res.st_mode)