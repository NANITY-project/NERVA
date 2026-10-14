# a small, fully-known GTK3 window for accessibility tests; writes each widget's allocation (window-relative) to a json file
import gi, json, sys, os
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, GLib
w = Gtk.Window(title="Nerva Test Window"); w.set_default_size(520, 360)
box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6, margin=10); w.add(box)
lbl = Gtk.Label(label="Hello accessibility world"); box.pack_start(lbl, False, False, 0)
btn = Gtk.Button(label="Save changes"); box.pack_start(btn, False, False, 0)
chk = Gtk.CheckButton(label="Enable feature"); chk.set_active(True); box.pack_start(chk, False, False, 0)
ent = Gtk.Entry(); ent.set_text("typed text here"); ent.get_accessible().set_name("Search box"); box.pack_start(ent, False, False, 0)
pw = Gtk.Entry(); pw.set_visibility(False); pw.set_text("hunter2-SECRET"); pw.get_accessible().set_name("Account password"); box.pack_start(pw, False, False, 0)
tv = Gtk.TextView(); tv.get_buffer().set_text("First line of notes\nSecond line of notes"); tv.get_accessible().set_name("Notes"); box.pack_start(tv, True, True, 0)
w.connect("destroy", Gtk.main_quit)
def dump():
    out = {}
    for name, wd in (("label", lbl), ("button", btn), ("check", chk), ("entry", ent), ("password", pw), ("notes", tv)):
        a = wd.get_allocation(); x, y = wd.translate_coordinates(w, 0, 0) or (0, 0); out[name] = [x, y, a.width, a.height]
    out["window"] = list(w.get_size()); out["pid"] = os.getpid()
    json.dump(out, open(sys.argv[1], "w")); return False
w.show_all(); GLib.timeout_add(1200, dump); Gtk.main()
