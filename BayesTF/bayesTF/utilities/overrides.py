#DKE Green
#2019


#https://stackoverflow.com/questions/1167617/in-python-how-do-i-indicate-im-overriding-a-method - for neatness
def overrides(interface_class):
    def overrider(method):
        assert(method.__name__ in dir(interface_class))
        return method
    return overrider

