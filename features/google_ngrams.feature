Scenario: Build a vector space from Google Books ngrams
    Given I have Google Books co-occurrence counts
    When I build a co-occurrence matrix
    And I evaluate the space on the wordsim353 dataset
    Then I should have the Google co-occurrence space file

Scenario: Apply tf-id weighting to the co-occurrence matrix
    Given I have Google Books co-occurrence counts
    When I build a co-occurrence matrix
    And I apply tf-idf weighting
    And I evaluate the space on the wordsim353 dataset
    Then I should have the tf-idf weighted space file


Scenario: Line-normalize the co-occurrence matrix
    Given I have Google Books co-occurrence counts
    When I build a co-occurrence matrix
    And I line-normalize the matrix
    And I evaluate the space on the wordsim353 dataset
    Then I should have the line-normalized space file


Scenario: Reduce the co-occurrence matrix using NMF
    Given I have Google Books co-occurrence counts
    When I build a co-occurrence matrix
    And I line-normalize the matrix
    And I apply NMF
    And I evaluate the space on the wordsim353 dataset
    Then I should have the reduced space file
