# vim: noet

ALTO_ENDPOINT = "https://xkcd.com/1975/alto/{}"

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from ctypes import cdll, c_int
import json
from pathlib import Path, PurePath
from queue import Queue
import re
import readline
import requests
from threading import Lock, Thread
import textwrap


class AltoStore:
	def __init__(self):
		self._store = {}
		self._executor = ThreadPoolExecutor()
	def get_delayed(self, k):
		if k not in self._store:
			self._store[k] = self._executor.submit(self._download_item, k)
		return self._store[k]
	def get(self, k):
		return self.get_delayed(k).result()
	def _download_item(self, k):
		return requests.get(ALTO_ENDPOINT.format(k)).json()
	def __del__(self):
		self._executor.shutdown(wait=False)


class Reaction:
	def __init__(self, entry, info):
		self.entry = entry
		self.menu = entry._menu
		if info:
			self.action = info['onAction']
		else:
			self.action = False
	def describe(self):
		return 'Do nothing'
	def enact(self, menu):
		if self.action:
			menu.perform_actions(self.action)
	@classmethod
	def create(cls, entry, info):
		"""create a Reaction from reaction json"""
		if info['tag'] == 'SubMenu':
			return Submenu(entry, info, info['subMenu'], info['subIdPostfix'])
		if info['tag'] == 'Action':
			if info['act'] is None:
				return Reaction(entry, info)
			if info['act']['tag'] == 'ColapseMenu':
				return Collapse(entry, info)
			if info['act']['tag'] == 'Nav':
				return Nav(entry, info, info['act']['url'])
		return UnknownAction(entry, info)


class Nav(Reaction):
	def __init__(self, entry, info, url):
		super().__init__(entry, info)
		self.url = url
	def describe(self):
		return f"Open <{self.url}>"
	def enact(self, menu):
		print(self.url)


class UnknownAction(Reaction):
	def __init__(self, entry, info):
		super().__init__(entry, info)
		self.info = info
	def describe(self):
		return f"Unknown action: {self.info!r}"
	def enact(self, menu):
		super().enact(menu)
		print(f"Unknown action: {self.info!r}")


class Collapse(Reaction):
	def describe(self):
		return "Reset the menu"
	def enact(self, menu):
		super().enact(menu)
		menu.open_top_menu()


class Submenu(Reaction):
	def __init__(self, entry, info, subid=False, subpostfix=False, item=False):
		super().__init__(entry, info)
		if subid is False and item is False:
			raise ValueError
		self._id = subid
		self._postfix = subpostfix
		self._item = item
	def describe(self):
		if self._item:
			return f"Open submenu {self._item.id}"
		if self._postfix:
			if not self._id:
				return f"Open submenu: (contents of tag {self.menu.display_tag(self._postfix)})"
			return f"Open submenu: {self._id} + contents of tag {self.menu.display_tag(self._postfix)}"
		return f"Open submenu {self._id}"
	@property
	def item(self):
		if self._item:
			return self._item
		id_ = self._id
		if self._postfix:
			if self._postfix not in self.menu._tags:
				return None
			id_ += self.menu._tags[self._postfix]
		return Item(self.menu, id_)
	def enact(self, menu):
		super().enact(menu)
		item = self.item
		parent = self.entry._item
		if parent and not self._postfix:
			hp = (parent.highest_parent[0] + 1, parent, self.entry)
			if hp[0] < item.highest_parent[0]:
				item.highest_parent = hp
		menu.open_submenu(self.entry, item)


class TagName:
	def __init__(self, tag):
		self.tag = tag


class Item:
	_id = None

	def __init__(self, menu, id_, path=None):
		with menu.new_item_lock:
			if self._id is not None:
				return
			self._id = id_
			self._menu = menu
			self.highest_parent = (float("inf"), None, None)
			if path is None:
				path = f"menu/{id_}"
			self._f = self._menu.store.get_delayed(path)

	def __new__(cls, menu, id_, path=None):
		if id_ in menu.items:
			return menu.items[id_]
		self = super().__new__(cls)
		menu.items[id_] = self
		return self

	def options(self):
		return [Entry(self._menu, x, self) for x in self.data['entries']]

	@property
	def id(self):
		return self._id

	@property
	def data(self):
		return self._f.result()

	def path_to(self):
		l = []
		x = self
		h = float("inf")
		while x:
			l.insert(0, x.highest_parent[2])
			h, x, _ = x.highest_parent
		if h > 0:
			raise ValueError("can't path here")
		return l

class Entry:
	def __init__(self, menu, info, item=None):
		self._menu = menu
		self._item = item
		self.data = info

	def path_to(self):
		if not self._item:
			return [self]
		return self._item.path_to() + [self]

	@property
	def display(self):
		return self.evaluate_condition(self._menu, self.data['display'])

	@property
	def active(self):
		return self.evaluate_condition(self._menu, self.data['active'])

	@property
	def label(self):
		return self.data['label']

	@property
	def reaction(self):
		return Reaction.create(self, self.data['reaction'])

	def __hash__(self):
		return hash((self._menu, self._item._id if self._item else None, self.label))

	def __eq__(self, other):
		return self._menu is other._menu and self._item == other._item and self.label == other.label

	def choose(self):
		self.reaction.enact(self._menu)

	def matches_selector(self, x):
		xc = x.casefold()
		l = self.label.casefold()
		if xc == l or xc == l.replace(' ', '_'):
			return True

	def __repr__(self):
		r = f"[{self.label}]"
		if not self.display:
			r += " hidden"
		elif not self.active:
			r += " inactive"
		return r

	def print_tag_effect(self, label, actions, cli):
		if not actions['setTags'] and not actions['unsetTags']:
			return
		print(f"{label}:")
		for k, v in actions['setTags'].items():
			cli.print_result(TagName(k), f"\tset {self._menu.display_tag(k)} to {v!r}")
		for v in actions['unsetTags']:
			cli.print_result(TagName(v), f"\tunset {self._menu.display_tag(v)}")

	def explain(self, cli):
		print(f"""{self!r}
display conditions: {Entry.format_condition(self._menu, self.data['display'])}
active conditions:  {Entry.format_condition(self._menu, self.data['active'])}
effect:             {self.reaction.describe()}""")
		oa = self.data['reaction']['onAction']
		self.print_tag_effect("tag effect", oa, cli)
		if isinstance(self.reaction, Submenu) and self.reaction.item:
			ol = self.reaction.item.data['onLeave']
			self.print_tag_effect("tag effect on leave", ol, cli)

	@classmethod
	def evaluate_condition(cls, menu, cond):
		if cond['tag'] == 'Always':
			return True
		if cond['tag'] == 'TagSet':
			return cond['contents'] in menu._tags
		if cond['tag'] == 'TagUnset':
			return cond['contents'] not in menu._tags
		if cond['tag'] == 'TLAnd':
			return all(cls.evaluate_condition(menu, x) for x in cond['contents'])
		if cond['tag'] == 'TLOr':
			return any(cls.evaluate_condition(menu, x) for x in cond['contents'])
		if cond['tag'] == 'TLNot':
			return not cls.evaluate_condition(menu, cond['contents'])
		raise ValueError

	@classmethod
	def format_condition(cls, menu, cond):
		if cond['tag'] == 'Always':
			return 'Always'
		if cond['tag'] == 'TagSet':
			return f"TagSet({menu.display_tag(cond['contents'])})={cond['contents'] in menu._tags}"
		if cond['tag'] == 'TagUnset':
			return f"TagUnset({menu.display_tag(cond['contents'])})={cond['contents'] not in menu._tags}"
		if cond['tag'] == 'TLAnd':
			return f"({' & '.join(cls.format_condition(menu, x) for x in cond['contents'])})"
		if cond['tag'] == 'TLOr':
			return f"({' | '.join(cls.format_condition(menu, x) for x in cond['contents'])})"
		if cond['tag'] == 'TLNot':
			return f"!{cls.format_condition(menu, cond['contents'])}"
		raise ValueError

	@classmethod
	def condition_deps(cls, cond):
		if cond['tag'] in ('TagSet', 'TagUnset'):
			return {cond['contents']}
		if cond['tag'] in ('TLAnd', 'TLOr'):
			return frozenset().union(*(cls.condition_deps(x) for x in cond['contents']))
		if cond['tag'] == 'TLNot':
			return cls.condition_deps(cond['contents'])
		return frozenset()


class FakeEntry(Entry):
	active = True
	display = True
	label = None
	reaction = None
	_item = None

	def __init__(self, menu, label, reaction):
		self._menu = menu
		self.label = label
		self.reaction = reaction

	def explain(self, cli):
		print(f"""{self!r}
effect:             {self.reaction.describe()}""")


class Walker:
	def __init__(self, menu):
		self.run = False
		self.running = False
		self.menu = menu
		self.items_seen = set()
		self.postfix_deps = defaultdict(set)
		self.possible_tag_values = defaultdict(set)
		self.explore_queue = Queue()
		self.workers = []
		# id -> (relative, entry that relates them)
		self.parents = defaultdict(set)
		self.children = defaultdict(set)
		# convenience: set of all entries
		self.entries = set()
		# for finding stuff out about tag values
		# tag -> (value, event name, event object)
		self.tag_events = defaultdict(set)
		# for finding what entries a tag influences
		# tag -> (what, entry)
		self.tag_deps = defaultdict(set)

	def _enqueue(self, id_, highest_parent):
		if id_ in self.items_seen:
			#print(f"already saw {id_}")
			self._update_parentage(id_, highest_parent)
			return
		self.items_seen.add(id_)
		#print(f"queueing {id_}")
		self.explore_queue.put((id_, highest_parent))

	def _update_parentage(self, item, highest_parent):
		item = Item(self.menu, item)
		if highest_parent[0] < item.highest_parent[0]:
			item.highest_parent = highest_parent

	def _add_child(self, entry, id_, root=False):
		parent = entry._item
		if root:
			highest_parent = (0, None, entry)
		elif parent:
			highest_parent = (parent.highest_parent[0] + 1, parent, entry)
		else:
			highest_parent = (float("inf"), None, None)
		self._enqueue(id_, highest_parent)
		if not parent:
			return
		self.parents[id_].add((parent._id, entry))
		self.children[parent._id].add((id_, entry))

	def _add_condition_deps(self, e, what, info):
		deps = Entry.condition_deps(info)
		for dep in deps:
			self.tag_deps[dep].add((what, e))

	def _add_from_entry(self, e, root=False):
		self.entries.add(e)
		self._add_condition_deps(e, 'display', e.data['display'])
		self._add_condition_deps(e, 'active', e.data['active'])
		r = e.reaction
		if r.action:
			self._add_actions(r.action, ('onAction', e))
		if not isinstance(r, Submenu):
			return
		if r._item:
			return
		if not r._postfix:
			self._add_child(e, r._id, root)
			return
		self.postfix_deps[r._postfix].add(r)
		ids = [r._id + v for v in self.possible_tag_values[r._postfix]]
		for id_ in ids:
			self._add_child(e, id_, root)

	def _add_tag_value(self, tag, value):
		if value in self.possible_tag_values[tag]:
			return
		self.possible_tag_values[tag].add(value)
		for r in self.postfix_deps[tag]:
			id_ = r._id + value
			self._add_child(r.entry, id_)

	def _add_actions(self, actions, event_data):
		for k, v in actions['setTags'].items():
			self._add_tag_value(k, v)
			self.tag_events[k].add((v, *event_data))
		for v in actions['unsetTags']:
			self.tag_events[v].add((None, *event_data))

	def _investigate(self, item):
		item, highest_parent = item
		self._update_parentage(item, highest_parent)
		item = Item(self.menu, item)
		#print(f"visiting {item.id}")
		self._add_actions(item.data['onLeave'], ('onLeave', item))
		ol = item.options()
		#print(f"{item.id}: considering {len(ol)} options")
		for opt in ol:
			self._add_from_entry(opt)

	def _worker(self):
		while True:
			e = self.explore_queue.get()
			if e is None:
				break
			try:
				self._investigate(e)
			finally:
				self.explore_queue.task_done()

	def explore(self):
		if self.running:
			return
		self.running = True
		for _ in range(20):
			t = Thread(target=self._worker)
			t.start()
			self.workers.append(t)
		for opt in self.menu.root_options():
			self._add_from_entry(opt, root=True)
		self.explore_queue.join()
		print("everything is done")
		for _ in self.workers:
			self.explore_queue.put(None)
		for w in self.workers:
			w.join()
		self.workers = []
		self.run = True


class Menu:
	def __init__(self):
		self.new_item_lock = Lock()
		self.store = AltoStore()
		self._tags = {}
		self.root = self.store.get('root')
		self.stack = []
		self.items = {}
		self.reset_tags()
		self.tag_names = {}
		self.tag_by_name = {}
		self.open_top_menu()
		try:
			self.load_tag_names()
		except OSError:
			pass

	def load_tag_names(self):
		with open('tags.json', 'r') as f:
			self.tag_by_name = json.load(f)
		self.tag_names = {v: k for k, v in self.tag_by_name.items()}

	def save_tag_names(self):
		with open('tags.json', 'w') as f:
			json.dump(self.tag_by_name, f)

	def root_options(self):
		return [Entry(self, menu) for menu in self.root['Menu']['entries']]

	def close(self):
		while self.stack:
			self.up()

	def open_top_menu(self):
		while self.stack:
			self.up()
		tm = self.get_top_menu()
		tm.choose()
		self.stack[-1][1].highest_parent = (0, None, tm)

	def get_top_menu(self):
		for menu in self.root['Menu']['entries']:
			entry = Entry(self, menu)
			if entry.display:
				return entry
		raise ValueError

	def display_tag(self, name, verbose=False):
		if verbose and name in self.tag_names:
			return f"{self.tag_names[name]}[{name}]"
		return self.tag_names.get(name, name)

	def reset_tags(self):
		self._tags = dict(self.root['State']['Tags'])

	def open_submenu(self, entry, item):
		self.stack.append((entry, item))

	def up(self):
		self.perform_actions(self.stack[-1][1].data['onLeave'])
		self.stack.pop(-1)

	def perform_actions(self, info):
		self._tags.update(info.get('setTags', {}))
		for k in info.get('unsetTags', []):
			if k in self._tags:
				del self._tags[k]

	def options(self):
		return self.stack[-1][1].options()


class Argument:
	NAME = 'argument'
	def usage(self):
		return self.NAME
	def convert(self, cli, x):
		return x
	def complete(self, cli, x):
		return []
	def _complete(self, cli, args, ai):
		return self.complete(cli, args[ai])
	def _consume(self, cli, l):
		"""returns a converted thing and a list of remaining args"""
		try:
			return [self.convert(cli, l[0])], l[1:]
		except IndexError:
			raise ArgumentError('missing required argument')


class ArgRecall(Argument):
	ACCEPTS = ()
	def _recall_value(self, cli, x):
		if not x.startswith('%'):
			return
		try:
			x = int(x[1:]) - 1
		except Exception:
			return
		try:
			return cli.results[x]
		except IndexError:
			return
	def _consume(self, cli, l):
		try:
			if l[0].startswith('%'):
				v = self._recall_value(cli, l[0])
				if isinstance(v, self.ACCEPTS):
					return [self.convert(cli, v)], l[1:]
				else:
					raise ArgumentError('incorrect type recalled')
			return super()._consume(cli, l)
		except IndexError:
			raise ArgumentError('missing required argument')


class ArgOptional(Argument):
	def __init__(self, arg, default=None):
		self.arg = arg
		self.default = default
	def usage(self):
		return f"[{self.arg.usage()}]"
	def convert(self, cli, x):
		return self.arg.convert(cli, x)
	def _complete(self, cli, args, ai):
		return self.arg._complete(cli, args, ai)
	def _consume(self, cli, l):
		if len(l) < 1:
			if self.default is not None:
				return self.arg._consume(cli, [self.default])
			return [None], []
		return self.arg._consume(cli, l)


class ArgUnion(Argument):
	def __init__(self, *args):
		self.args = args
	def usage(self):
		return "{%s}" % ' | '.join(a.usage() for a in self.args if a.usage())
	def convert(self, cli, x):
		for arg in self.args:
			try:
				return arg.convert(cli, x)
			except ArgumentError:
				continue
		raise ArgumentError(f"expected one of: {', '.join(a.usage() for a in self.args)}")
	def complete(self, cli, x):
		return [v for l in (a.complete(cli, x) for a in self.args) for v in l]
	def _consume(self, cli, l):
		for arg in self.args:
			try:
				return arg._consume(cli, l)
			except ArgumentError:
				continue
		raise ArgumentError(f"expected one of: {', '.join(a.usage() for a in self.args)}")


class ArgSeq(Argument):
	def __init__(self, *args):
		self.args = args
	def usage(self):
		u = []
		for param in self.args:
			pu = param.usage()
			if pu:
				u.append(pu)
		return ' '.join(u)
	def _complete(self, cli, args, ai):
		ac = len(args)
		cli.cdprint(f"ai = {ai}, ac = {ac}")
		for param in self.args:
			cli.cdprint(f"{param!r}: pre : {args}")
			try:
				_, ca = param._consume(cli, args)
			except ArgumentError:
				ca = []
			cli.cdprint(f"{param!r}: post: {ca}")
			if ac - len(ca) > ai:
				ai -= ac - len(args)
				cli.cdprint(f"final ai: {ai}")
				break
			args = ca
		return param._complete(cli, args, ai)
	def _consume(self, cli, l):
		al = []
		for param in self.args:
			v, l = param._consume(cli, l)
			al.extend(v)
		return al, l


class ArgWord(Argument):
	NAME = 'STR'


class ArgAll(Argument):
	NAME = 'STR...'
	def _consume(self, cli, l):
		return [self.convert(cli.menu, ' '.join(l))], []


class ArgPath(ArgAll):
	NAME = 'PATH...'
	def convert(self, cli, x):
		return PurePath(x)


class ArgInt(Argument):
	NAME = 'INTEGER'
	def convert(self, cli, x):
		try:
			return int(x)
		except Exception:
			raise ArgumentError(f"can't interpret {x!r} as an integer")


class ArgConst(Argument):
	NAME = ''
	def __init__(self, constant, argument=None):
		self.constant = constant
		self.argument = argument
	def _consume(self, cli, l):
		if self.argument:
			return [self.argument.convert(cli, self.constant)], l
		else:
			return [self.constant], l


class ArgLiteral(Argument):
	UNSET = object()
	def __init__(self, literal, value=UNSET, ignore_case=True):
		self.literal = literal
		self.value = [] if value is self.UNSET else [value]
		self.ignore_case = ignore_case
	def usage(self):
		return f'"{self.literal}"'
	def convert(self, cli, x):
		if x == self.literal:
			return self.value
		if self.ignore_case and x.casefold() == self.literal.casefold():
			return self.value
		raise ArgumentError(f"expected {self.literal}")
	def complete(self, cli, x):
		if self.literal.startswith(x):
			return [self.literal]
		if self.ignore_case and self.literal.casefold().startswith(x.casefold()):
			return [self.literal]
		return []
	def _consume(self, cli, l):
		try:
			return self.convert(cli, l[0]), l[1:]
		except IndexError:
			raise ArgumentError('missing required argument')


class ArgTag(ArgRecall):
	NAME = 'TAG'
	ACCEPTS = (TagName,)
	def convert(self, cli, x):
		if isinstance(x, TagName):
			return x.tag
		if x in cli.menu.tag_names:
			return x
		if x in cli.menu.tag_by_name:
			return cli.menu.tag_by_name[x]
		if cli.walker.run and x not in cli.walker.tag_events:
			raise ArgumentError(f"tag {x!r} does not exist")
		return x
	def complete(self, cli, x):
		if cli.walker.run:
			all_known_tags = set(cli.walker.tag_events.keys()) | set(cli.menu.tag_names.values())
		else:
			all_known_tags = set(cli.menu._tags.keys()) | set(cli.menu.tag_names.keys()) | set(cli.menu.tag_names.values())
		return sorted([t for t in all_known_tags if t.startswith(x)])


class ArgExistingTag(ArgTag):
	def convert(self, cli, x):
		x = super().convert(cli, x)
		if x not in cli.menu._tags:
			raise ArgumentError(f"tag {x!r} not set")
		return x
	def complete(self, cli, x):
		all_known_tags = set(cli.menu._tags.keys()) | set(v for k, v in cli.menu.tag_names.items() if k in cli.menu._tags)
		return sorted([t for t in all_known_tags if t.startswith(x)])


class ArgSubmenu(ArgRecall):
	NAME = 'SUBMENU'
	ACCEPTS = (Entry,)
	def convert(self, cli, x):
		if isinstance(x, Entry):
			return x
		if isinstance(x, Item):
			if x.highest_parent[2] is not None:
				return x.highest_parent[2]
			else:
				raise ArgumentError(f"selecting this thing is not supported")
		if x.isdigit():
			n = False
			try:
				n = int(x)
			except Exception:
				pass
			if n and n > 0 and n <= len(cli.menu.stack[-1][1].options()):
				return cli.menu.stack[-1][1].options()[n-1]
		if x == '.':
			return cli.menu.stack[-1][0]
		else:
			for e in cli.menu.stack[-1][1].options():
				if e.matches_selector(x):
					return e
		raise ArgumentError(f"no such submenu {x!r}")
	def complete(self, cli, x):
		cl = []
		if '.'.startswith(x):
			cl.append('.')
		for e in cli.menu.stack[-1][1].options():
			l = e.label.casefold().replace(' ', '_')
			if l.startswith(x.casefold()):
				cl.append(e.label.replace(' ', '_'))
		return cl


class ArgCommand(Argument):
	NAME = 'COMMAND'
	def convert(self, cli, x):
		if not x.startswith('-'):
			x = '-' + x
		x = x[1:].casefold()
		try:
			cli.get_command(x)
			return x
		except KeyError:
			raise ArgumentError("no such command")
	def complete(self, cli, x):
		if not x.startswith('-'):
			x = '-' + x
		x = x[1:].casefold()
		commands = sorted(_Commands.commands.keys())
		return [f"-{cmd}" for cmd in commands if cmd.startswith(x)]


class CommandError(Exception):
	def __init__(self, message):
		self.message = message
	def __str__(self):
		return self.message


class ArgumentError(CommandError):
	pass


class Command:
	ARGS = []
	REQUIRES_FULL_DATA = False

	def __init__(self, cli, menu):
		self.cli = cli
		self.menu = menu
		self.args = ArgSeq(*self.ARGS)

	def complete(self, args, ai):
		args = args.split(' ')
		return self.args._complete(self.cli, args, ai)

	def _execute(self, args):
		cmd, *args = args
		try:
			al, sl = self.args._consume(self.cli, args)
			if sl:
				raise ArgumentError("too many arguments")
		except ArgumentError as e:
			print(e)
			print(f"-{cmd} {self.usage()}")
			return
		if self.REQUIRES_FULL_DATA and not self.cli.walker.run:
			print('this command requires a full dump of the menu; -walk first')
			return
		try:
			return self.execute(*al)
		except CommandError as e:
			print(f"error: {e}")
		except Exception as e:
			print(f"got {e!r} while running command")

	def usage(self):
		return self.args.usage()


class _Commands:
	class help(Command):
		"""
		Print usage and help text for a command, or list all the commands
		"""
		ARGS = [ArgUnion(ArgLiteral("list", None), ArgCommand())]
		def execute(self, cmd):
			if cmd is None:
				print(' '.join(sorted(f"-{x}" for x in _Commands.commands.keys())))
				return
			command = self.cli.get_command(cmd)
			print(f"-{cmd} {command.usage()}")
			if command.__doc__:
				print(textwrap.dedent(command.__doc__).strip('\n'))
	class dump(Command):
		"""
		Save everything we know to the filesystem somewhere
		"""
		ARGS = [ArgPath()]
		#REQUIRES_FULL_DATA = True
		def execute(self, path):
			if '..' in path.parts:
				raise ArgumentError("don't be silly")
			p = Path.cwd() / path
			try:
				p.mkdir()
			except FileExistsError:
				raise ArgumentError("target already exists")
			for item, data in self.menu.store._store.items():
				ip = p / item
				ip.parent.mkdir(parents=True, exist_ok=True)
				with ip.open('w') as f:
					json.dump(data.result(), f)
			print(f"dumped menu at {p.resolve().as_uri()}")
	class _complete(Command):
		"""
		Debug tab-completion for some text
		"""
		ARGS = [ArgAll()]
		def execute(self, text):
			args = text.split(' ')
			self.cli.completion_debug = True
			print(self.cli.completions(args[-1], len(text) - len(args[-1]), len(text) - 1, text))
			self.cli.completion_debug = False
		def complete(self, args, ai):
			al = args.split(' ')
			return self.cli.completions(al[ai], len(' '.join(al[:ai])), len(' '.join(al[:ai + 1])), args)
	class walk(Command):
		"""
		Walk the entire menu, remembering various things for other processing
		"""
		def execute(self):
			print("this will take a while...")
			self.cli.walker.explore()
	class all_tags(Command):
		"""
		List all the known tags
		"""
		REQUIRES_FULL_DATA = True
		def execute(self):
			for tag in sorted(self.cli.walker.tag_events.keys()):
				if tag in self.menu._tags:
					print(f"{self.menu.display_tag(tag, True)} = {self.menu._tags[tag]!r}")
				else:
					print(f"{self.menu.display_tag(tag, True)} (not set)")
	class explain_tag(Command):
		"""
		List menu items which affect a given tag
		"""
		ARGS = [ArgTag()]
		REQUIRES_FULL_DATA = True
		def helpful_path(self, e):
			p = e.path_to()
			if p == [e]:
				return "(root menu)"
			else:
				return ' > '.join(e.label for e in entry.path_to())
		def execute(self, tag):
			for ev in self.cli.walker.tag_events[tag]:
				value, etype, eobj = ev
				verb = f"set to {value!r}" if value is not None else "unset"
				what = "unknown"
				if etype == 'onAction':
					what = f"clicking [{eobj.label}]"
				elif etype == 'onLeave':
					what = f"leaving menu {eobj.id}"
				self.cli.print_result(eobj, f"{verb} by {what}")
				print(f"\tat: {self.helpful_path(eobj)}")
			for what, entry in self.cli.walker.tag_deps[tag]:
				self.cli.print_result(entry, f"controls {what} of [{entry.label}]")
				print(f"\tat: {self.helpful_path(entry)}")
	class routes_to(Command):
		"""
		List all the distinct, non-cyclic routes to a given menu
		"""
		ARGS = [ArgOptional(ArgSeq(ArgUnion(ArgSubmenu(), ArgWord()), ArgOptional(ArgInt())), default='.')]
		REQUIRES_FULL_DATA = True
		def execute(self, item, depth):
			start = []
			if isinstance(item, str):
				item = Item(self.menu, item)
			if isinstance(item, Entry):
				r = item.reaction
				if isinstance(r, Submenu):
					item = r.item
				else:
					item = item._item
					start = [item]
			if item is None:
				print("at top level")
				return
			if depth is not None and depth < 1:
				raise ArgumentError("depth can't be less than 1")
			routes = []
			def explore(item, route, seen=frozenset()):
				if item in seen: # this is a cycle
					return
				seen = seen | {item}
				io = Item(self.menu, item)
				if io.highest_parent[0] == 0:
					routes.append([io.highest_parent[2]] + route)
					return
				parents = self.cli.walker.parents[item]
				for parent, entry in parents:
					explore(parent, [entry] + route, seen)
			explore(item.id, start)
			if depth is not None:
				rs = set(tuple(route[-depth:]) for route in routes)
				routes = [route[0].path_to() + list(route[1:]) for route in rs]
			labels = [[e.label for e in route] for route in routes]
			for rl in sorted(labels):
				print(' > '.join(rl))
	class choose(Command):
		"""
		Navigate to a child menu
		"""
		ARGS = [ArgSubmenu()]
		def execute(self, opt):
			opt.choose()
	class ls(Command):
		"""
		List current menu entries
		"""
		ARGS = [ArgOptional(ArgLiteral("all", True))]
		def execute(self, all_):
			for opt in self.cli.options:
				if not all_ and not opt.display:
					continue
				if not opt.active:
					self.cli.print_result(opt, f"[{opt.label}]")
				else:
					self.cli.print_result(opt, opt.label)
	class goto(Command):
		"""
		Navigate to a menu, starting at the top
		"""
		ARGS = [ArgSubmenu()]
		REQUIRES_FULL_DATA = True
		def execute(self, entry):
			self.menu.close()
			for e in entry.path_to():
				e.choose()
	class jump(Command):
		"""
		Navigate directly to a menu by ID by means of a synthetic menu item
		"""
		ARGS = [ArgUnion(ArgSubmenu(), ArgWord())]
		def execute(self, item):
			if isinstance(item, str):
				item = Item(self.menu, item)
			elif isinstance(item, Entry):
				r = item.reaction
				if not isinstance(r, Submenu):
					raise CommandError("that can't be jumped to")
				item = r.item
			entry = FakeEntry(self.menu, f"Jump to {id_}", None)
			entry.reaction = Submenu(entry, None, item=item)
			entry.choose()
	class find(Command):
		"""
		Search all menu items for a regular expression
		"""
		ARGS = [ArgAll()]
		REQUIRES_FULL_DATA = True
		def execute(self, text):
			regexp = re.compile(text, re.I)
			for opt in self.cli.walker.entries:
				if not regexp.search(opt.label):
					continue
				self.cli.print_result(opt, ' > '.join(e.label for e in opt.path_to()))
	class explain(Command):
		"""
		Print general information about a menu item
		"""
		ARGS = [ArgOptional(ArgSubmenu(), default='.')]
		def execute(self, submenu):
			submenu.explain(self.cli)
	class tags(Command):
		"""
		List the tags currently in effect
		"""
		def execute(self):
			for k, v in self.menu._tags.items():
				self.cli.print_result(TagName(k), f"{self.menu.display_tag(k, True)} = {v!r}")
	class rename_tag(Command):
		"""
		Set or change the friendly name for a tag
		"""
		ARGS = [ArgTag(), ArgWord()]
		def execute(self, tag, name):
			if tag in self.menu.tag_names:
				old = self.menu.tag_names[tag]
				del self.menu.tag_by_name[old]
			self.menu.tag_names[tag] = name
			self.menu.tag_by_name[name] = tag
			self.menu.save_tag_names()
	class set_tag(Command):
		"""
		Set the value of a tag
		"""
		ARGS = [ArgTag(), ArgAll()]
		def execute(self, tag, value):
			self.menu._tags[tag] = value
			print(f"Set tag {self.menu.display_tag(tag)} to {value!r}")
	class unset_tag(Command):
		"""
		Clear a tag
		"""
		ARGS = [ArgExistingTag()]
		def execute(self, tag):
			if tag not in self.menu._tags:
				print("Not set")
			else:
				del self.menu._tags[tag]
			print(f"Unset {self.menu.display_tag(tag)}")
	class reset(Command):
		"""
		Reset all tags and open the top-level menu
		"""
		def execute(self):
			while self.menu.stack:
				self.menu.up()
			self.menu.reset_tags()
			self.menu.open_top_menu()
	class path(Command):
		"""
		Print the path we took to the current menu
		"""
		def execute(self):
			print(' > '.join(e[0].label for e in self.menu.stack))
	class shortest_path(Command):
		"""
		Print the shortest known path to the current menu
		"""
		ARGS = [ArgOptional(ArgWord())]
		def execute(self, arg):
			l = []
			if arg:
				x = Item(self.menu, arg)
			else:
				x = self.menu.stack[-1][1]
			print(' > '.join(e.label for e in x.path_to()))
	class up(Command):
		"""
		Navigate up one level
		"""
		def execute(self):
			if len(self.menu.stack) < 2:
				print("at top level")
			else:
				self.menu.up()
	class close(Command):
		"""
		Open the top-level menu
		"""
		def execute(self):
			self.menu.open_top_menu()
	class ls_root(Command):
		"""
		List the root menu entries
		"""
		def execute(self):
			root_options = [Entry(self.menu, info) for info in self.menu.root['Menu']['entries']]
			for opt in root_options:
				self.cli.print_result(opt, opt.label)
	class open_root(Command):
		"""
		Open a synthetic menu containing the root menu entries (main menu, power, etc.)
		"""
		def execute(self):
			class RootItem:
				id = '(root menu)'
				options = lambda _: [Entry(self.menu, info) for info in self.menu.root['Menu']['entries']]
				data = {'onLeave': {'setTags': {}, 'unsetTags': []}}
			item = RootItem()
			entry = FakeEntry(self.menu, 'Root', None)
			entry.reaction = Submenu(entry, None, item=item)
			while self.menu.stack:
				self.menu.up()
			self.menu.stack.append((entry, item))
	class _eval(Command):
		"""
		Evaluate some Python
		"""
		ARGS = [ArgAll()]
		def execute(self, code):
			v = eval(code)
			if v is not None:
				print(v)


_Commands.commands = {k.replace('_', '-'): v for k, v in _Commands.__dict__.items() if not k.startswith('__')}


class CLI:
	def __init__(self):
		self.menu = Menu()
		self.walker = Walker(self.menu)
		self._commands = {}
		self.completion_debug = False
		self.last_stack = []
		self.results = []
		self.reset_results = False

	def cdprint(self, x):
		if self.completion_debug:
			print(x)

	def print_result(self, v, x):
		if self.reset_results:
			self.reset_results = False
			self.results = []
		print(f"{len(self.results) + 1:3}: {x}")
		self.results.append(v)

	def get_command(self, name):
		if name not in self._commands:
			self._commands[name] = _Commands.commands[name](self, self.menu)
		return self._commands[name]

	def run(self):
		readline.set_completer(self._completer)
		while True:
			self.prompt_and_run_once()

	def select_opt(self, t_):
		t = t_.casefold()
		display = True
		if t.startswith('.'):
			t = t[1:]
			display = False
		for opt in self.options:
			if display and not opt.display:
				continue
			if opt.label.casefold() == t:
				opt.choose()
				return
		if t:
			print(f"Unknown choice: {t_}")

	@property
	def options(self):
		return self.menu.options()

	@property
	def prompt(self):
		return f"[{self.menu.stack[-1][0].label}] "

	def completions(self, s_, b, e, line):
		s = s_.casefold()
		if line.startswith('-'):
			if b > 0:
				cmd, args = line[1:].split(' ', 1)
				if cmd not in _Commands.commands:
					return []
				ai = line[:b].count(' ') - 1
				return self.get_command(cmd).complete(args, ai)
			cl = [f"-{x}" for x in _Commands.commands.keys()]
		elif line.startswith('.'):
			cl = ['..'] + [f".{opt.label}" for opt in self.options if not opt.display]
		else:
			cl = [opt.label for opt in self.options if opt.display and opt.active]
		return sorted([x for x in cl if x.casefold().startswith(s)])

	def _completer(self, s, n):
		ck = (readline.get_line_buffer(), readline.get_begidx())
		if self._ct != ck:
			self._ct = ck
			b, e = readline.get_begidx(), readline.get_endidx()
			buf = readline.get_line_buffer()
			self._cl = self.completions(s, b, e, buf)
		return self._cl[n] if n < len(self._cl) else None

	def prompt_and_run_once(self):
		self.reset_results = True
		if self.menu.stack != self.last_stack:
			print('')
			for opt in self.options:
				if not opt.display:
					continue
				if not opt.active:
					print(f"[{opt.label}]")
				else:
					print(opt.label)
		self.last_stack = self.menu.stack[:]
		self.dont_print_menu = False
		self._ct = None
		cmd = input(self.prompt)
		if cmd == '..':
			cmd = '-up'
		if cmd.startswith('-'):
			self.dont_print_menu = True
			args = cmd[1:].split(' ')
			cmd = args[0]
			if cmd not in _Commands.commands:
				print("no such command")
			else:
				self.get_command(cmd)._execute(args)
		else:
			self.select_opt(cmd)


def unfuck_readline():
	libreadline = cdll.LoadLibrary("libreadline.so")
	c_int.in_dll(libreadline, "rl_sort_completion_matches").value = 0


def main():
	unfuck_readline()
	readline.parse_and_bind("tab: menu-complete")
	readline.set_completer_delims(" ")

	c = CLI()
	c.run()

if __name__ == '__main__':
	main()
