class Item:
    def __init__(self, _id, title, genres, score=None):
        self.id = _id
        self.title = title
        self.score = score
        self.genres = genres

    def __repr__(self):
        return self.title


def create_item_mapping(df_item, item_col, title_col, genre_col):
    """Create a dictionary of item id to Item lookup."""
    item_mapping = {}
    for row in df_item.itertuples():
        item_id = getattr(row, item_col)
        item_title = getattr(row, title_col)
        item_genres = getattr(row, genre_col)



        # Check if 'categories' is a list
        if isinstance(item_genres, list):
            genres_list = item_genres

        else:
            genres_list = item_genres.split('|')

        genre_ratio = 1. / len(genres_list)
        item_genre = {genre: genre_ratio for genre in genres_list}

        item = Item(item_id, item_title, item_genre)
        item_mapping[item_id] = item

    return item_mapping
