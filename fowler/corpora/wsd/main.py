"""Implementation of common word sense disambiguation methods."""
import logging

import pandas as pd

from fowler.corpora.dispatcher import Dispatcher, Resource, SpaceMixin
from fowler.corpora.models import read_space_from_file
from fowler.corpora.util import display

from .experiments import SimilarityExperiment
from .datasets import KS13Dataset


logger = logging.getLogger(__name__)


class WSDDispatcher(Dispatcher, SpaceMixin):
    """WSD task dispatcher."""

    global__composition_operator = (
        '',
        (
            'verb',
            'add',
            'mult',
            'kron',
            'relational',
            'copy-object',
            'copy-subject',
            'frobenious-add',
            'frobenious-mult',
            'frobenious-outer',
        ),
        'Composition operator.',
    )
    global__google_vectors = '', False, 'Get rid of certain words in the input data that are not in Google vectors.'
    global__verb_space = '', '', 'Separate transitive verb space.'

    @Resource
    def verb_space(self):
        if self.kwargs['verb_space']:
            return read_space_from_file(self.kwargs['verb_space'])

    @Resource
    def gs11_data(self):
        """The data set grouped by verb, subject, object and landmark.

        The mean of the input values per group is calculated.

        """
        data = pd.read_csv(self.kwargs['gs11_data'], sep=' ')
        grouped = data.groupby(
            ('verb', 'subject', 'object', 'landmark', 'hilo'),
            as_index=False,
        ).mean()

        if self.google_vectors:
            grouped = grouped[grouped['landmark'] != 'mope']
            grouped = grouped[grouped['object'] != 'behaviour']
            grouped = grouped[grouped['object'] != 'favour']
            grouped = grouped[grouped['object'] != 'offence']
            grouped = grouped[grouped['object'] != 'paper']

        if self.limit:
            grouped = grouped.head(self.limit)

        return grouped

    @Resource
    def gs12_data(self):
        """The data set grouped by 'adj_subj', 'subj', 'verb', 'landmark', 'adj_obj', 'obj'.

        The mean of the input values per group is calculated.

        """
        index_cols = 'adj_subj', 'subj', 'verb', 'landmark', 'adj_obj', 'obj'

        data = pd.read_csv(
            self.kwargs['gs12_data'],
            sep=' ',
            usecols=index_cols + ('annotator_score', ),
        )
        grouped = data.groupby(index_cols, as_index=False).mean()

        grouped['obj'][grouped['obj'] == 'papers'] = 'paper'

        if self.google_vectors:
            grouped = grouped[grouped['obj'] != 'offence']
            grouped = grouped[grouped['obj'] != 'favour']

        if self.limit:
            grouped = grouped.head(self.limit)

        return grouped


dispatcher = WSDDispatcher()
command = dispatcher.command


def gs11_similarity(args):
    (v, verb), (s, subject), (o, object_), (l, landmark), composition_operator = args

    if composition_operator == 'kron':
        subject_object = compose(subject, object_)

        sentence_verb = verb.multiply(subject_object)
        sentence_landmark = landmark.multiply(subject_object)
    elif composition_operator == 'mult':
        sentence_verb = verb.multiply(subject).multiply(object_)
        sentence_landmark = landmark.multiply(subject).multiply(object_)
    elif composition_operator == 'add':
        sentence_verb = verb + subject + object_
        sentence_landmark = landmark + subject + object_
    elif composition_operator == 'verb':
        sentence_verb = verb
        sentence_landmark = landmark

    return pairwise.cosine_similarity(sentence_verb, sentence_landmark)[0][0]


@command()
def gs11(
    pool,
    space,
    composition_operator,
    gs11_data=('', 'downloads/compdistmeaning/GS2011data.txt', 'The GS2011 dataset.'),
):
    """Categorical compositional distributional model for transitive verb disambiguation.

    Implements method described in [1]. The data is available at [2].

    [1] Grefenstette, Edward, and Mehrnoosh Sadrzadeh. "Experimental support
    for a categorical compositional distributional model of meaning."
    Proceedings of the Conference on Empirical Methods in Natural Language
    Processing. Association for Computational Linguistics, 2011.

    [2] http://www.cs.ox.ac.uk/activities/compdistmeaning/GS2011data.txt

    """
    similarity_experiment(
        space,
        pool,
        gs11_data,
        verb_columns=('verb', 'landmark'),
        similarity_input=lambda verb_vectors, t: (
            (
                (v, verb_vectors[v]),
                (s, space[t.S(s)]),
                (o, space[t.S(o)]),
                (l, verb_vectors[l]),
                composition_operator,
            )
            for v, s, o, l in gs11_data[['verb', 'subject', 'object', 'landmark']].values
        ),
        similarity_function=gs11_similarity,
        input_column='input',
        composition_operator=composition_operator,
    )

    display(gs11_data.groupby('hilo').mean())


@command()
def sentence_similarity(
        pool,
        no_p11n,
        composition_operator,
        space,
        verb_space,
        dataset=('', 'downloads/compdistmeaning/emnlp2013_turk.txt', 'The KS2013 dataset.'),
        tagset=('', ('bnc', 'bnc+ccg', 'ukwac'), 'Space tagset'),
        output=('o', 'sentence_similarity.h5', 'Result output file.'),
        no_group=('', False, "Don't calculate the mean of human judgments."),
        human_judgement_column=('', 'score', 'Column name for human judgments.'),
):
    common_kwargs = {
        'show_progress_bar': not no_p11n,
        'pool': pool,
    }

    experiment = SimilarityExperiment(**common_kwargs)

    experiment.evaluate(
        KS13Dataset(
            dataset_filename=dataset,
            composition_operator=composition_operator,
            space=space,
            verb_space=verb_space,
            tagset=tagset,
            group=not no_group,
            human_judgement_column=human_judgement_column,
            **common_kwargs
        ),
        composition_operator=composition_operator,
    ).to_hdf(output)


def gs12_similarity(args):
    (
        (as_, adj_subj),
        (s, subj),
        (v, verb),
        (l, landmark),
        (ao, adj_obj),
        (o, obj),
        composition_operator,
        np_composition,
    ) = args

    def compose(a, n):
        return {
            'add': lambda: a + n,
            'mult': lambda: a.multiply(n),
        }[np_composition]()

    return gs11_similarity(
        (
            (v, verb), (s, compose(adj_subj, subj)), (o, compose(adj_obj, obj)), (l, landmark), composition_operator,
        )
    )


@command()
def gs12(
    pool,
    space,
    composition_operator,
    np_composition=('', 'mult', 'Operation used to compose adjective with noun. [add|mult]'),
    gs12_data=('', 'downloads/compdistmeaning/GS2012data.txt', 'The GS2012 dataset.'),
):
    similarity_experiment(
        space,
        pool,
        gs12_data,
        verb_columns=('verb', 'landmark'),
        similarity_input=lambda verb_vectors, t: (
            (
                (as_, space[t.A(as_)]),
                (s, space[t.S(s)]),
                (v, verb_vectors[v]),
                (l, verb_vectors[l]),
                (ao, space[t.A(ao)]),
                (o, space[t.S(o)]),
                composition_operator,
                np_composition,
            )
            for as_, s, v, l, ao, o in gs12_data[['adj_subj', 'subj', 'verb', 'landmark', 'adj_obj', 'obj']].values
        ),
        similarity_function=gs12_similarity,
        input_column='annotator_score',
        composition_operator=composition_operator,
    )


@command()
def ks13_targets(
    dataset=('', 'downloads/compdistmeaning/emnlp2013_turk.txt', 'The KS2013 dataset.'),
    out=('o', 'targets.csv', 'KS targets'),
    tagset=('', 'bnc', 'Tagset'),
):
    """Extract target words from the EMNLP2013_turk dataset."""

    df = KS13Dataset.read(dataset)

    tag_mappings = {
        'bnc': {'N': 'SUBST', 'V': 'VERB'},
        'bnc+ccg': {'N': 'N', 'V': 'V'},
        'ukwac': {'N': 'N', 'V': 'V'},
    }

    def extract_tokens(frame, tag=None):
        frame = frame.unique()
        result = pd.DataFrame({'ngram': frame})

        if tag is not None:
            result['tag'] = tag

        return result

    tm = tag_mappings[tagset]


    (
       pd
       .concat(
           [
               extract_tokens(df[c], t)
               for c, t in (
                   ('verb1', tm['V']), ('subject1', tm['N']), ('object1', tm['N']),
                   ('verb2', tm['V']), ('subject2', tm['N']), ('object2', tm['N']),
               )
           ]
        )
       .drop_duplicates()
       .to_csv(out, index=False)
    )
