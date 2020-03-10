from DataClasses import *
from FilesIO import *
from multiprocessing import Lock, cpu_count
from multiprocessing.pool import ThreadPool
import itertools as it
import wikipediaapi


class Miner:
    def __init__(self, root_page, max_level, load_from_file, file_name, level_1_categories=None,
                 level_2_categories=None, level_3_categories=None):
        print("Miner: starts")
        if load_from_file:
            print("Miner: loading data from JSON file...")
            self.data = load_JSON_data(file_name + ".json")
        else:
            self.root_page_title = root_page.lower()
            self.main_topic_categories = wikipediaapi.Wikipedia('en').page(self.root_page_title).categorymembers
            self.level_1_categories = level_1_categories
            self.level_2_categories = level_2_categories
            self.level_3_categories = level_3_categories
            self.data = None
            self.__init_data()
            print("Miner: mining data...")
            self.mine(max_level=max_level)
            print("Miner: saving data to JSON file...")
            save_JSON_data(self.data, file_name + ".json", beautify=True)
        print("Miner: finished")

    def mine(self, max_level=1, max_sub_categories=10, max_articles_per_category=1000):
        self.__init_data()
        self.__traverse_multiprocessors(category_members=self.main_topic_categories,
                                        curr_level=1, max_level=max_level, max_sub_categories=max_sub_categories,
                                        max_articles_per_category=max_articles_per_category)
        self.__add_ids_to_data_members()
        self.__convert_dictionaries_keys_to_ids()
        self.__build_categories_ids_by_levels_array(max_level)
        self.__find_descendant_articles_for_categories()
        self.__add_direct_category_id_to_articles()
        self.__convert_categories_descendant_articles_sets_to_lists()

    def __init_data(self):
        self.data = Data()
        # create & add the wikipedia root category:
        self.data.categories[self.root_page_title] = Category(self.root_page_title, None, 0)

    def add_top_level_categories_and_articles(self, members):
        self.data.categories[self.root_page_title] = Category(self.root_page_title, None, 0)
        for member in members:
            if member.ns == wikipediaapi.Namespace.CATEGORY:
                # member is a category
                category = Category(member.title, self.root_page_title, 1)
                self.data.add_category(category)
                self.data.categories[self.root_page_title].add_subcategory(category.title)
            elif member.ns == wikipediaapi.Namespace.MAIN:
                # member is an article:
                article = Article(member)
                self.data.add_article(article)
                self.data.categories[self.root_page_title].add_article(article.title)

    def __traverse_multiprocessors(self, category_members, curr_level=1, max_level=1,
                                   max_sub_categories=4, max_articles_per_category=None):
        category_members_titles = [member.title() for member in category_members]
        top_level_categories_list = []
        for category_title in self.level_1_categories:
            if category_title in category_members_titles:
                top_level_categories_list.append(category_members[category_title])
            else:
                category = wikipediaapi.Wikipedia('en').page(category_title)
                category.ns = wikipediaapi.Namespace.CATEGORY
                top_level_categories_list.append(category)
        top_level_categories_list = top_level_categories_list[:min(len(top_level_categories_list), max_sub_categories)]
        top_level_articles = [member for member in category_members.values() if
                              member.ns == wikipediaapi.Namespace.MAIN]
        top_level_articles = top_level_articles[:min(len(top_level_articles), max_articles_per_category)]
        self.add_top_level_categories_and_articles(top_level_categories_list + top_level_articles)

        categories_names = [member.title.lower() for member in top_level_categories_list]
        num_categories = len(categories_names)
        categories_members_list = [member.categorymembers for member in top_level_categories_list]
        parameters = list(zip(categories_names, categories_members_list, it.repeat(curr_level + 1, num_categories),
                              it.repeat(max_level, num_categories), it.repeat(max_sub_categories, num_categories),
                              it.repeat(max_articles_per_category, num_categories)))
        global lock
        lock = Lock()
        max_simultaneous_threads = cpu_count()  # TODO: unused variable, remove this line ?
        pool = ThreadPool(processes=10)
        pool.starmap(self.traverse_categories_tree_synchronized, parameters)
        pool.close()
        pool.join()

    def traverse_categories_tree_synchronized(self, category_name, category_members, curr_level=1, max_level=1,
                                              max_sub_categories=4,
                                              max_articles_per_category=None):
        # adding a few sub-categories manually:
        if category_name == "category:health":
            physical_exercise_category = wikipediaapi.Wikipedia('en').page("Category:Physical exercise")
            physical_exercise_category.ns = wikipediaapi.Namespace.CATEGORY
            category_members[physical_exercise_category.title] = physical_exercise_category
        elif category_name == "category:technology":
            computers_category = wikipediaapi.Wikipedia('en').page("Category:Computers")
            computers_category.ns = wikipediaapi.Namespace.CATEGORY
            category_members[computers_category.title] = computers_category

        added_sub_categories_counter = 0
        added_articles_counter = 0
        for member in category_members.values():
            if member.ns == wikipediaapi.Namespace.CATEGORY and curr_level <= max_level:  # member is a category
                if added_sub_categories_counter < max_sub_categories and self.should_mine_member(member.title):
                    if curr_level == 2 and member.title in self.level_2_categories or \
                            curr_level == 3 and member.title in self.level_3_categories:
                        added_sub_categories_counter += 1
                        category = Category(member.title, category_name, curr_level)
                        lock.acquire()
                        self.data.add_category(category)
                        self.data.categories[category_name].add_subcategory(category.title)
                        lock.release()
                        self.traverse_categories_tree_synchronized(member.title.lower(), member.categorymembers,
                                                                   curr_level + 1,
                                                                   max_level,
                                                                   max_sub_categories, max_articles_per_category)
                    elif curr_level == 3:
                        self.traverse_categories_tree_synchronized(category_name, member.categorymembers,
                                                                   curr_level + 1,
                                                                   max_level,
                                                                   max_sub_categories, max_articles_per_category=10)
            elif member.ns == wikipediaapi.Namespace.MAIN:  # member is an article
                if added_articles_counter < max_articles_per_category and self.should_mine_member(member.title):
                    added_articles_counter += 1
                    article = Article(member)
                    lock.acquire()
                    self.data.add_article(article)
                    self.data.categories[category_name].add_article(article.title)
                    lock.release()
            if added_sub_categories_counter >= max_sub_categories and \
                    added_articles_counter >= max_articles_per_category:
                return

    def should_mine_member(self, title):
        title = title.lower()
        if title == 'category:clothes in politics':
            return True
        filter_list = ["index of", "outline of", " by ", "list of", " lists", "associated with",
                       "comparisons", " stubs", " in ", "Template:"]
        for filter_word in filter_list:
            if filter_word in title:
                return False
        return True

    def __add_ids_to_data_members(self):
        id = 0
        for category in self.data.categories.values():
            category.id = str(id)
            self.data.categories_ids.add(category.id)
            id += 1

        for article in self.data.articles.values():
            article.id = str(id)
            id += 1

        for category in self.data.categories.values():
            for article in category.articles:
                category.direct_articles_ids.append(self.data.articles[article].id)
            for sub_category in category.sub_categories:
                category.sub_categories_ids.append(self.data.categories[sub_category].id)
            if category.parent:
                category.parent_id = self.data.categories[category.parent].id

        for article in self.data.articles.values():
            for category in article.categories_titles:
                if category in self.data.categories:
                    article.categories_ids.add(self.data.categories[category].id)

    def __convert_dictionaries_keys_to_ids(self):
        categories_by_id = {}
        for category in self.data.categories.values():
            categories_by_id[category.id] = category
        self.data.categories = categories_by_id

        articles_by_id = {}
        for article in self.data.articles.values():
            articles_by_id[article.id] = article
        self.data.articles = articles_by_id

    def __build_categories_ids_by_levels_array(self, max_level):
        self.data.categories_ids_by_levels = [[] for _ in range(max_level + 1)]
        for category in self.data.categories.values():
            self.data.categories_ids_by_levels[category.level].append(category.id)

    def __find_descendant_articles_for_categories(self, category_id='0'):
        descendant_articles = set(self.data.categories[category_id].direct_articles_ids)
        for sub_category_id in self.data.categories[category_id].sub_categories_ids:
            articles = self.__find_descendant_articles_for_categories(sub_category_id)
            descendant_articles = descendant_articles.union(articles)
        self.data.categories[category_id].descendant_articles = descendant_articles
        return descendant_articles

    def __add_direct_category_id_to_articles(self):
        for category in self.data.categories.values():
            for article_id in category.direct_articles_ids:
                self.data.articles[str(article_id)].categories_ids.add(category.id)

    def __convert_categories_descendant_articles_sets_to_lists(self):
        for category in self.data.categories.values():
            category.descendant_articles = list(category.descendant_articles)
