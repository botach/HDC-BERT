import csv
from nltk.tokenize import sent_tokenize


class StatisticsGenerator:
    def __init__(self, data, statistics_by_level_file_name, statistics_by_category_file_name):
        print("StatisticsGenerator: starts")
        self.data = data
        self.statistics_by_level_file_name = statistics_by_level_file_name
        self.statistics_by_category_file_name = statistics_by_category_file_name
        self.get_statistics_by_level()
        self.get_statistics_by_category()
        print("StatisticsGenerator: finished")

    def get_statistics_by_category(self):
        statistics_by_category = {}
        for category_id in self.data.categories_ids_by_levels[1]:
            category = self.data.categories[category_id]
            category_title = category.title.replace('category:', '').title()
            statistics_by_category[category_title] = {}
            statistics_by_category[category_title]["Number of Articles"] = len(category.descendant_articles)
            full_text_word_counter = 0
            summary_word_counter = 0
            first_sentence_word_counter = 0
            for article_id in category.descendant_articles:
                article = self.data.articles[article_id]
                full_text_word_counter += len(article.summary.split())
                summary_word_counter += len(article.summary.split())
                first_sentence = sent_tokenize(article.summary)[0]
                first_sentence_word_counter += len(first_sentence.split())
                for section in article.sections:
                    full_text_word_counter += len(section[1].split())
            avg_full_text_word_count = full_text_word_counter/len(category.descendant_articles)
            avg_summary_word_count = summary_word_counter/len(category.descendant_articles)
            avg_first_sentence_word_count = first_sentence_word_counter/len(category.descendant_articles)
            statistics_by_category[category_title]["Average Article Number of Words"] = float(avg_full_text_word_count)
            statistics_by_category[category_title]["Average Summary Number of Words"] = float(avg_summary_word_count)
            statistics_by_category[category_title]["Average First Sentence Number of Words"] = float(avg_first_sentence_word_count)

        fields = ["Category", "Number of Articles", "Average Article Number of Words", "Average Summary Number of Words",
                  "Average First Sentence Number of Words"]
        with open(self.statistics_by_category_file_name, "w") as f:
            w = csv.writer(f)
            headers = fields[1:]
            w.writerow(fields)
            for key in statistics_by_category.keys():
                w.writerow([key] + [round(statistics_by_category[key][h],1) for h in headers])
        return statistics_by_category

    def get_statistics_by_level(self):
        statistics_by_level = {}
        for level in range(1, 4):
            statistics_by_level[level] = {}
            statistics_by_level[level]["Number of Articles"] = 0
            statistics_by_level[level]["Average Article Number of Words"] = 0
            statistics_by_level[level]["Average Summary Number of Words"] = 0
            statistics_by_level[level]["Average First Sentence Number of Words"] = 0
            for category_id in self.data.categories_ids_by_levels[level]:
                category = self.data.categories[category_id]
                statistics_by_level[level]["Number of Articles"] += len(category.direct_articles_ids)
                full_text_word_counter = 0
                summary_word_counter = 0
                first_sentence_word_counter = 0
                for article_id in category.direct_articles_ids:
                    article = self.data.articles[article_id]
                    full_text_word_counter += len(article.summary.split())
                    summary_word_counter += len(article.summary.split())
                    first_sentence = sent_tokenize(article.summary)[0]
                    first_sentence_word_counter += len(first_sentence.split())
                    for section in article.sections:
                        full_text_word_counter += len(section[1].split())
                avg_full_text_word_count = full_text_word_counter/len(category.direct_articles_ids)
                avg_summary_word_count = summary_word_counter/len(category.direct_articles_ids)
                avg_first_sentence_word_count = first_sentence_word_counter/len(category.direct_articles_ids)
                statistics_by_level[level]["Average Article Number of Words"] += float(avg_full_text_word_count)
                statistics_by_level[level]["Average Summary Number of Words"] += float(avg_summary_word_count)
                statistics_by_level[level]["Average First Sentence Number of Words"] += float(avg_first_sentence_word_count)
            statistics_by_level[level]["Average Article Number of Words"] /= len(self.data.categories_ids_by_levels[level])
            statistics_by_level[level]["Average Summary Number of Words"] /= len(self.data.categories_ids_by_levels[level])
            statistics_by_level[level]["Average First Sentence Number of Words"] /= len(self.data.categories_ids_by_levels[level])

        fields = ["Level", "Number of Articles", "Average Article Number of Words", "Average Summary Number of Words",
                  "Average First Sentence Number of Words"]
        with open(self.statistics_by_level_file_name, "w") as f:
            w = csv.writer(f)
            headers = fields[1:]
            w.writerow(fields)
            for key in statistics_by_level.keys():
                w.writerow([key] + [round(statistics_by_level[key][h],1) for h in headers])
        return statistics_by_level
