class RepetitionCheck:
    def __init__(self, rp_length: int):
        self.rp_length = rp_length

    def __call__(self, token_ids, logprobs):
        # check if the token_ids are repeated at the last rp_length (at least) tokens
        if len(token_ids) < 2 * self.rp_length:
            return False
        # find the appearance of the last token
        last_token = token_ids[-1]
        import numpy as np
        last_index = np.where(token_ids == last_token)[0]
        if last_index.size < 2:
            return False
        for index in last_index[:-1]:
            if index > self.rp_length:
                str1 = token_ids[index - self.rp_length: index]
                str2 = token_ids[-self.rp_length:]
                # Compare the lists element by element since they are numpy arrays
                if np.array_equal(str1, str2):
                    return True
        return False
