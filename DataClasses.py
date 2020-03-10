# The random seed used by all non-deterministic procedures in the project
RANDOM_SEED = 10


class Category:
    def __init__(self, title, parent, level):
        self.title = title.lower()
        self.id = None
        self.level = level
        self.parent = parent
        self.parent_id = None
        self.sub_categories = []
        self.sub_categories_ids = []
        self.articles = []
        self.direct_articles_ids = []
        self.descendant_articles = set()

    def add_subcategory(self, sub_category):
        self.sub_categories.append(sub_category.lower())

    def add_article(self, article):
        self.articles.append(article.lower())


class Article:
    def __init__(self, article):
        self.title = article.title.lower()
        self.id = None
        self.summary = article.summary
        self.sections = self.extract_sections(article.sections)
        self.categories_titles = [category.title().lower() for category in article.categories]
        self.categories_ids = set()
        self.vectors = {}

    def extract_sections(self, original_sections):
        sections = []
        for section in original_sections:
            if self.__should_mine_section(section.title.lower()):
                sections.append((section.title.lower(), section.text))
        return sections

    def __should_mine_section(self, section_title):
        filter_list = ["references", "further reading", "external links"]
        for filter_word in filter_list:
            if filter_word in section_title:
                return False
        return True


class Data:
    def __init__(self):
        self.categories = {}
        self.categories_ids = set()
        self.articles = {}
        self.categories_ids_by_levels = []

    def add_category(self, category):
        if category.title not in self.categories.keys():
            print("adding category: " + category.title)
            self.categories[category.title] = category

    def add_article(self, article):
        if article.title not in self.articles.keys():
            print("adding article: " + article.title)
            self.articles[article.title] = article
